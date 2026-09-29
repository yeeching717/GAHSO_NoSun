"""SFC 需求與流量產生器；資源範圍依 Table I，分布假設見 README。"""
from dataclasses import dataclass
from math import isfinite
from random import Random


@dataclass(frozen=True)
class VNF:
    """鏈上單一虛擬功能，cpu 的單位為論文的 CPU unit。"""
    name: str
    cpu: float

    def __post_init__(self):
        if not self.name or not isfinite(self.cpu) or self.cpu <= 0:
            raise ValueError("VNF needs a name and positive finite CPU demand")


@dataclass(frozen=True)
class SFCRequest:
    """不可變請求：vnfs 的順序就是資料必須依序處理的順序。"""
    id: str
    source: int
    destination: int
    vnfs: tuple[VNF, ...]
    bandwidth_mhz: float
    max_delay_s: float
    arrival_s: float
    duration_s: float
    service_type: str = "custom"

    def __post_init__(self):
        object.__setattr__(self, "vnfs", tuple(self.vnfs))
        if not self.id or len(self.vnfs) < 2 or not all(isinstance(v, VNF) for v in self.vnfs):
            raise ValueError("Request needs an ID and at least source/destination VNFs")
        if type(self.source) is not int or type(self.destination) is not int or min(self.source, self.destination) < 0:
            raise ValueError("Endpoints must be nonnegative integer node IDs")
        if not all(isfinite(x) and x > 0 for x in (self.bandwidth_mhz, self.max_delay_s, self.duration_s)):
            raise ValueError("Bandwidth, delay limit and duration must be finite and positive")
        if not isfinite(self.arrival_s) or self.arrival_s < 0 or not isfinite(self.expires_s):
            raise ValueError("Invalid arrival/expiry time")

    @property
    def expires_s(self) -> float:
        """固定生命週期截止時間；資源配置不會重新延長服務期限。"""
        return self.arrival_s + self.duration_s


@dataclass(frozen=True)
class RequestConfig:
    """流量參數；頻寬型／計算型比值 0.5 對應頻寬型機率 1/3。"""
    arrival_rate_per_slot: float = 3.0
    slot_duration_s: float = 10.0
    mean_duration_s: float = 120.0
    bandwidth_to_compute_ratio: float = 0.5
    vnf_count_min: int = 3
    vnf_count_max: int = 5
    cpu_min: float = 50.0
    cpu_max: float = 200.0
    bandwidth_min_mhz: float = 20.0
    bandwidth_max_mhz: float = 80.0
    max_delay_s: float = 0.100
    seed: int = 11

    def __post_init__(self):
        positive = (self.slot_duration_s, self.mean_duration_s, self.cpu_min, self.cpu_max,
                    self.bandwidth_min_mhz, self.bandwidth_max_mhz, self.max_delay_s)
        if not all(isfinite(x) and x > 0 for x in positive):
            raise ValueError("Request configuration must be finite and positive")
        if not isfinite(self.arrival_rate_per_slot) or self.arrival_rate_per_slot < 0:
            raise ValueError("Arrival rate must be finite and nonnegative")
        if not isfinite(self.bandwidth_to_compute_ratio) or self.bandwidth_to_compute_ratio < 0:
            raise ValueError("Service ratio must be finite and nonnegative")
        if (type(self.vnf_count_min) is not int or type(self.vnf_count_max) is not int
                or not 2 <= self.vnf_count_min <= self.vnf_count_max):
            raise ValueError("Require 2 <= vnf_count_min <= vnf_count_max")
        if self.cpu_min > self.cpu_max or self.bandwidth_min_mhz > self.bandwidth_max_mhz:
            raise ValueError("Invalid demand ranges")


class RequestGenerator:
    def __init__(self, config: RequestConfig | None = None):
        self.config = config or RequestConfig()

    def generate(self, horizon_s: float, satellite_count: int) -> list[SFCRequest]:
        """使用指數到達間隔產生 Poisson 流量，同一組設定會重現相同請求。"""
        if not isfinite(horizon_s) or horizon_s <= 0 or satellite_count < 2:
            raise ValueError("Positive horizon and at least two satellites required")
        cfg = self.config
        rng = Random(cfg.seed)  # 每次呼叫從相同種子開始，方便比較不同策略。
        if cfg.arrival_rate_per_slot == 0:
            return []
        rate_per_second = cfg.arrival_rate_per_slot / cfg.slot_duration_s
        p_bandwidth = cfg.bandwidth_to_compute_ratio / (1 + cfg.bandwidth_to_compute_ratio)
        requests, time_s = [], 0.0
        cpu_mid = (cfg.cpu_min + cfg.cpu_max) / 2
        bw_mid = (cfg.bandwidth_min_mhz + cfg.bandwidth_max_mhz) / 2
        while True:
            time_s += rng.expovariate(rate_per_second)
            if time_s >= horizon_s:
                break
            is_bandwidth = rng.random() < p_bandwidth
            service_type = "bandwidth" if is_bandwidth else "compute"
            # 補充假設：將 Table I 的需求區間從中點分割為兩類服務。
            cpu_range = (cfg.cpu_min, cpu_mid) if is_bandwidth else (cpu_mid, cfg.cpu_max)
            bw_range = (bw_mid, cfg.bandwidth_max_mhz) if is_bandwidth else (cfg.bandwidth_min_mhz, bw_mid)
            source, destination = rng.sample(range(satellite_count), 2)
            vnfs = tuple(VNF(f"vnf-{i}", rng.uniform(*cpu_range))
                         for i in range(rng.randint(cfg.vnf_count_min, cfg.vnf_count_max)))
            requests.append(SFCRequest(
                f"sfc-{len(requests):06d}", source, destination, vnfs, rng.uniform(*bw_range),
                cfg.max_delay_s, time_s, max(1e-9, rng.expovariate(1 / cfg.mean_duration_s)), service_type,
            ))
        return requests
