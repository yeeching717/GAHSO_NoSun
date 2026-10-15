"""圓軌道 Walker Star 星座：位置用 km、時間用秒；對應論文公式 (1)–(3)。"""
from dataclasses import dataclass, field
from math import cos, sin, sqrt, pi, radians, isfinite
from random import Random
from types import MappingProxyType
from typing import Mapping

EARTH_RADIUS_KM = 6371.0
EARTH_MU_KM3_S2 = 398600.4418
EARTH_ROTATION_RAD_S = 7.2921159e-5
LIGHT_SPEED_KM_S = 299000.0  # Paper Table I: 2.99e8 m/s.
Position = tuple[float, float, float]
Edge = tuple[int, int]


def edge_key(a: int, b: int) -> Edge:
    """無向鏈路使用統一鍵，避免正反方向被當作兩份容量。"""
    return (min(a, b), max(a, b))


def distance(a: Position, b: Position) -> float:
    return sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def line_of_sight(a: Position, b: Position, radius: float) -> bool:
    """計算線段到地心的最近距離，排除穿越地球的鏈路。"""
    delta = tuple(y - x for x, y in zip(a, b))
    length_sq = sum(x * x for x in delta)
    if length_sq == 0:
        return False
    u = max(0.0, min(1.0, -sum(x * y for x, y in zip(a, delta)) / length_sq))
    closest_sq = sum((x + u * y) ** 2 for x, y in zip(a, delta))
    return closest_sq >= radius * radius - 1e-6


@dataclass(frozen=True)
class ConstellationConfig:
    """預設 25 顆＝5 軌道×5 顆；名義四鄰居與實際通視連線分開。"""
    satellites: int = 25
    planes: int = 5
    altitude_km: float = 2000.0
    inclination_deg: float = 85.0
    phase_factor: int = 0
    cpu_min: float = 400.0
    cpu_max: float = 600.0
    link_bandwidth_mhz: float = 200.0
    earth_radius_km: float = EARTH_RADIUS_KM
    earth_mu_km3_s2: float = EARTH_MU_KM3_S2
    light_speed_km_s: float = LIGHT_SPEED_KM_S
    enforce_line_of_sight: bool = True
    close_plane_seam: bool = True
    seed: int = 7

    def __post_init__(self):
        if any(type(v) is not int for v in (self.satellites, self.planes, self.phase_factor)):
            raise ValueError("Satellite count, plane count and phase factor must be integers")
        if self.planes < 3 or self.satellites % self.planes or self.satellites // self.planes < 3:
            raise ValueError("Require >=3 planes, >=3 satellites per plane, and equal plane populations")
        if not 0 <= self.phase_factor < self.planes:
            raise ValueError("phase_factor must be in [0, planes)")
        positive = (self.altitude_km, self.cpu_min, self.cpu_max, self.link_bandwidth_mhz,
                    self.earth_radius_km, self.earth_mu_km3_s2, self.light_speed_km_s)
        if not all(isfinite(x) and x > 0 for x in positive) or self.cpu_min > self.cpu_max:
            raise ValueError("Capacities and physical constants must be finite and positive")
        if not isfinite(self.inclination_deg) or not 0 <= self.inclination_deg <= 180:
            raise ValueError("Inclination must be between 0 and 180 degrees")


@dataclass(frozen=True)
class Satellite:
    id: int
    plane: int
    slot: int
    position_km: Position
    cpu_capacity: float


@dataclass(frozen=True)
class Link:
    a: int
    b: int
    distance_km: float
    delay_s: float
    capacity_mhz: float
    kind: str


@dataclass(frozen=True)
class NetworkSnapshot:
    """不可變拓樸快照；鄰接表只包含當下存在且可通視的鏈路。"""
    time_s: float
    satellites: Mapping[int, Satellite]
    links: Mapping[Edge, Link]
    nominal_links: Mapping[Edge, Link] = field(default_factory=dict)
    _neighbors: Mapping[int, tuple[int, ...]] = field(init=False, repr=False)

    def __post_init__(self):
        satellites, links = dict(self.satellites), dict(self.links)
        adjacent = {n: [] for n in satellites}
        for a, b in links:
            adjacent[a].append(b)
            adjacent[b].append(a)
        object.__setattr__(self, "satellites", MappingProxyType(satellites))
        object.__setattr__(self, "links", MappingProxyType(links))
        object.__setattr__(self, "nominal_links", MappingProxyType(dict(self.nominal_links)))
        object.__setattr__(self, "_neighbors",
                           MappingProxyType({n: tuple(sorted(v)) for n, v in adjacent.items()}))

    def neighbors(self, node: int) -> tuple[int, ...]:
        return self._neighbors[node]


@dataclass(frozen=True)
class GroundTerminal:
    latitude_deg: float
    longitude_deg: float
    minimum_elevation_deg: float = 0.0

    def __post_init__(self):
        if not all(isfinite(x) for x in (self.latitude_deg, self.longitude_deg, self.minimum_elevation_deg)):
            raise ValueError("Terminal coordinates must be finite")
        if not -90 <= self.latitude_deg <= 90 or not -180 <= self.longitude_deg <= 180:
            raise ValueError("Invalid geographic coordinates")
        if not 0 <= self.minimum_elevation_deg <= 90:
            raise ValueError("Minimum elevation must be in [0, 90]")

    def nearest_visible(self, snapshot: NetworkSnapshot, radius_km: float = EARTH_RADIUS_KM) -> int | None:
        # 將地固座標依地球自轉轉換為慣性座標，與衛星位置使用相同座標系。
        lat = radians(self.latitude_deg)
        lon = radians(self.longitude_deg) + EARTH_ROTATION_RAD_S * snapshot.time_s
        ground = (radius_km * cos(lat) * cos(lon), radius_km * cos(lat) * sin(lon), radius_km * sin(lat))
        choices = []
        for n, sat in snapshot.satellites.items():
            d = distance(ground, sat.position_km)
            if d == 0:
                continue
            elevation_sine = sum((s - g) * g / radius_km for s, g in zip(sat.position_km, ground)) / d
            if elevation_sine >= sin(radians(self.minimum_elevation_deg)) - 1e-12:
                choices.append((d, n))
        return min(choices)[1] if choices else None


class LEOConstellation:
    """固定硬體容量、隨時間重算位置與鏈路；相同 seed/time 可重現結果。"""
    def __init__(self, config: ConstellationConfig | None = None):
        self.config = config or ConstellationConfig()
        rng = Random(self.config.seed)
        self.cpu = tuple(rng.uniform(self.config.cpu_min, self.config.cpu_max)
                         for _ in range(self.config.satellites))

    @property
    def orbital_period_s(self) -> float:
        cfg = self.config
        return 2 * pi * sqrt((cfg.earth_radius_km + cfg.altitude_km) ** 3 / cfg.earth_mu_km3_s2)

    def snapshot(self, time_s: float, disabled_links=()) -> NetworkSnapshot:
        """以公式 (2) 計算位置、公式 (1) 計算延遲，並限制最多四條 ISL。"""
        if not isfinite(time_s) or time_s < 0:
            raise ValueError("time_s must be finite and nonnegative")
        cfg = self.config
        count = cfg.satellites // cfg.planes
        radius = cfg.earth_radius_km + cfg.altitude_km
        inclination = radians(cfg.inclination_deg)
        angular_speed = 2 * pi / self.orbital_period_s
        satellites = {}
        for n in range(cfg.satellites):
            plane, slot = divmod(n, count)
            # Walker Star：升交點赤經在 180 度內均分；相位與角速度決定軌道位置。
            raan = pi * plane / cfg.planes
            anomaly = 2 * pi * slot / count + 2 * pi * cfg.phase_factor * plane / cfg.satellites
            anomaly += angular_speed * time_s
            position = (
                radius * (cos(anomaly) * cos(raan) - sin(anomaly) * cos(inclination) * sin(raan)),
                radius * (cos(anomaly) * sin(raan) + sin(anomaly) * cos(inclination) * cos(raan)),
                radius * sin(anomaly) * sin(inclination),
            )
            satellites[n] = Satellite(n, plane, slot, position, self.cpu[n])
        links, nominal = {}, {}
        disabled = {edge_key(*edge) for edge in disabled_links}

        def candidate(a, b):
            if edge_key(a, b) in disabled:
                return None
            pa, pb = satellites[a].position_km, satellites[b].position_km
            d = distance(pa, pb)
            # 極區交會的重合節點不建立零距離鏈路；視線仍須不被地球遮蔽。
            if d < 1e-6 or (cfg.enforce_line_of_sight and not line_of_sight(pa, pb, cfg.earth_radius_km)):
                return None
            return d

        def add(a, b, d, kind):
            key = edge_key(a, b)
            link = Link(*key, d, d / cfg.light_speed_km_s, cfg.link_bandwidth_mhz, kind)
            nominal[key] = link
            if candidate(a, b) is not None:
                links[key] = link

        for plane in range(cfg.planes):
            for slot in range(count):
                a, b = plane * count + slot, plane * count + (slot + 1) % count
                add(a, b, distance(satellites[a].position_km, satellites[b].position_km), "intra")
        pairs = [(p, p + 1) for p in range(cfg.planes - 1)]
        if cfg.close_plane_seam:
            pairs.append((cfg.planes - 1, 0))
        for left, right in pairs:
            a_ids = list(range(left * count, (left + 1) * count))
            b_ids = list(range(right * count, (right + 1) * count))
            costs = [[distance(satellites[a].position_km, satellites[b].position_km)
                      for b in b_ids] for a in a_ids]
            # 名義配對先決定，遮蔽或故障只移除邊，不另補邊湊度數。
            for i, j in enumerate(minimum_distance_matching(costs)):
                add(a_ids[i], b_ids[j], costs[i][j], "inter")
        return NetworkSnapshot(time_s, satellites, links, nominal)


def minimum_distance_matching(costs):
    """Hungarian assignment: minimum total distance, deterministic ties, O(n^3).

    Each satellite gets one counterpart in each adjacent plane; this is not
    independent nearest-neighbor selection, which can violate reciprocity.
    """
    n = len(costs)
    if not n or any(len(row) != n or any(not isfinite(x) or x < 0 for x in row) for row in costs):
        raise ValueError("A finite nonnegative square cost matrix is required")
    u, v, p, way = [0.] * (n+1), [0.] * (n+1), [0] * (n+1), [0] * (n+1)
    for i in range(1, n+1):
        p[0], j0 = i, 0
        minv, used = [float('inf')] * (n+1), [False] * (n+1)
        while True:
            used[j0] = True
            i0, delta, j1 = p[j0], float('inf'), 0
            for j in range(1, n+1):
                if not used[j]:
                    cur = costs[i0-1][j-1] - u[i0] - v[j]
                    if cur < minv[j]:
                        minv[j], way[j] = cur, j0
                    if minv[j] < delta:
                        delta, j1 = minv[j], j
            for j in range(n+1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while j0:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
    result = [0] * n
    for j in range(1, n+1):
        result[p[j]-1] = j-1
    return result
