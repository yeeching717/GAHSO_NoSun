"""論文公式 (4)–(13)：負載、公平性、C1–C8 檢查與原子資源帳本。"""
from collections import Counter
from dataclasses import dataclass, replace
from enum import Enum
from math import isfinite
from typing import Iterable
from .network import Edge, NetworkSnapshot, edge_key
from .services import SFCRequest

EPSILON = 1e-8


class ServiceStatus(str, Enum):
    """服務從接受、暫時中斷，到到期或主動釋放的生命週期。"""
    ACTIVE = "active"
    INTERRUPTED = "interrupted"
    REJECTED = "rejected"
    EXPIRED = "expired"
    RELEASED = "released"


@dataclass(frozen=True)
class Allocation:
    """path 為簡單實體路徑；vnf_nodes[i] 為第 i 個 VNF 所在衛星。"""
    path: tuple[int, ...]
    vnf_nodes: tuple[int, ...]
    endpoint_placement_mode: str = "fixed"

    def __post_init__(self):
        object.__setattr__(self, "path", tuple(self.path))
        object.__setattr__(self, "vnf_nodes", tuple(self.vnf_nodes))
        if any(type(n) is not int for n in self.path + self.vnf_nodes):
            raise ValueError("Allocation node IDs must be integers")
        if self.endpoint_placement_mode not in ("fixed", "neighbor_allowed"):
            raise ValueError("Unknown endpoint placement mode")


@dataclass(frozen=True)
class Decision:
    accepted: bool
    reason: str


@dataclass
class ServiceRecord:
    request: SFCRequest
    status: ServiceStatus
    allocation: Allocation | None = None
    reason: str = ""
    interrupted_since_s: float | None = None
    interruption_count: int = 0


@dataclass(frozen=True)
class ConstraintCheck:
    """一項約束的檢查結果；reason 是穩定的機器可讀失敗代碼。"""
    code: str
    passed: bool
    reason: str = ""


@dataclass(frozen=True)
class FeasibilityReport:
    """只讀診斷，不配置資源；checks 可同時指出多項失敗原因。"""
    checks: tuple[ConstraintCheck, ...]
    delay_s: float | None

    @property
    def feasible(self) -> bool:
        return all(check.passed for check in self.checks)

    @property
    def violations(self) -> tuple[str, ...]:
        return tuple(check.reason for check in self.checks if not check.passed)

    @property
    def reason(self) -> str:
        return next(iter(self.violations), "")


def jain_fairness(loads: Iterable[float]) -> float:
    """公式 (5)；先縮放避免平方溢位。全空載定義為 1，空網路拒絕計算。"""
    values = tuple(loads)
    if not values or any(not isfinite(x) or x < 0 for x in values):
        raise ValueError("Loads must be a nonempty sequence of finite nonnegative values")
    scale = max(values)
    if scale == 0:
        return 1.0
    values = tuple(value / scale for value in values)
    return sum(values) ** 2 / (len(values) * sum(value * value for value in values))


class ResourceManager:
    """單執行緒帳本；候選配置先驗證，成功後才一次扣除資源。"""
    def __init__(self, snapshot: NetworkSnapshot, cpu_weight: float = 1.0, bandwidth_weight: float = 1.0):
        if (not all(isfinite(x) and x >= 0 for x in (cpu_weight, bandwidth_weight))
                or cpu_weight + bandwidth_weight <= 0):
            raise ValueError("Load weights must be finite, nonnegative and not both zero")
        self.snapshot = snapshot
        self.cpu_weight = cpu_weight
        self.bandwidth_weight = bandwidth_weight
        self.records: dict[str, ServiceRecord] = {}
        self._cpu_used: Counter[int] = Counter()
        self._bandwidth_used: Counter[Edge] = Counter()
        self.events: list[dict] = []

    def planning_copy(self, release_request_id: str | None = None):
        """候選搜索使用獨立帳本；可只在副本歸還待遷移服務自己的資源。"""
        other = ResourceManager(self.snapshot, self.cpu_weight, self.bandwidth_weight)
        other.records = {key: replace(record) for key, record in self.records.items()}
        other._cpu_used = self._cpu_used.copy()
        other._bandwidth_used = self._bandwidth_used.copy()
        if release_request_id is not None:
            record = other.records.get(release_request_id)
            if record is None or record.status not in (ServiceStatus.ACTIVE, ServiceStatus.INTERRUPTED):
                raise ValueError("Only a reserved service can receive migration credit")
            other._apply(record.request, record.allocation, -1)
            del other.records[release_request_id]
        other.assert_invariants()
        return other

    @property
    def time_s(self):
        return self.snapshot.time_s

    def cpu_available(self, node: int) -> float:
        return self.snapshot.satellites[node].cpu_capacity - self._cpu_used[node]

    def bandwidth_available(self, a: int, b: int) -> float:
        key = edge_key(a, b)
        link = self.snapshot.links.get(key)
        return 0.0 if link is None else link.capacity_mhz - self._bandwidth_used[key]

    def _event(self, kind: str, request_id: str, reason: str = ""):
        self.events.append({"time_s": self.time_s, "event": kind, "request_id": request_id, "reason": reason})

    @staticmethod
    def _demand(request: SFCRequest, allocation: Allocation):
        cpu = Counter()
        bandwidth = Counter()
        for vnf, node in zip(request.vnfs, allocation.vnf_nodes):
            cpu[node] += vnf.cpu
        for a, b in zip(allocation.path, allocation.path[1:]):
            bandwidth[edge_key(a, b)] += request.bandwidth_mhz
        return cpu, bandwidth

    def check_allocation(self, request: SFCRequest, allocation: Allocation) -> FeasibilityReport:
        """公開的可行性檢查：C1–C8 與實作的簡單路徑／VNF 順序限制。"""
        report = self._check_allocation(request, allocation)
        if request.id in self.records:
            return FeasibilityReport(
                (ConstraintCheck("REQUEST_ID", False, "duplicate_request_id"),) + report.checks,
                report.delay_s,
            )
        return report

    def _check_allocation(self, request: SFCRequest, allocation: Allocation,
                          credit: Allocation | None = None) -> FeasibilityReport:
        # credit 僅供既有服務替換配置：自己占用的資源可以回收後重新配置。
        path, placements = allocation.path, allocation.vnf_nodes
        checks = []

        def add(code, passed, reason):
            checks.append(ConstraintCheck(code, bool(passed), "" if passed else reason))

        add("ARRIVAL", self.time_s >= request.arrival_s, "request_not_arrived")
        add("EXPIRY", self.time_s < request.expires_s, "request_already_expired")
        endpoints = bool(path) and path[0] == request.source and path[-1] == request.destination
        simple = len(set(path)) == len(path)
        known = all(n in self.snapshot.satellites for n in path + placements)
        add("PATH", endpoints, "invalid_path_endpoints")
        add("SIMPLE_PATH", simple, "path_contains_cycle")
        add("NODES", known, "unknown_satellite")
        add("C3", len(placements) == len(request.vnfs), "vnf_count_mismatch")
        if allocation.endpoint_placement_mode == "fixed":
            first_ok = bool(placements) and placements[0] == request.source
            last_ok = bool(placements) and placements[-1] == request.destination
        else:
            first_ok = bool(placements) and placements[0] in ({request.source} | set(self.snapshot.neighbors(request.source)))
            last_ok = bool(placements) and placements[-1] in ({request.destination} | set(self.snapshot.neighbors(request.destination)))
        add("C1", first_ok, "invalid_vnf_endpoints")
        add("C2", last_ok, "invalid_vnf_endpoints")
        indices = {node: i for i, node in enumerate(path)}
        on_path = all(n in indices for n in placements)
        positions = [indices[n] for n in placements] if on_path else []
        ordered = on_path and positions == sorted(positions)
        add("VNF_PATH", on_path, "vnf_outside_path")
        add("VNF_ORDER", ordered, "vnf_order_violation")
        edges = [edge_key(a, b) for a, b in zip(path, path[1:])]
        linked = all(key in self.snapshot.links for key in edges)
        add("LINKS", linked, "link_unavailable")
        delay = sum(self.snapshot.links[key].delay_s for key in edges) if linked and path else None
        add("C4", delay is not None and delay <= request.max_delay_s + 1e-12, "delay_exceeded")

        # C5：每對相鄰 VNF 的子路徑，出流－入流須等於其起終點的嵌入差。
        flow_ok = endpoints and simple and known and linked and ordered and len(placements) == len(request.vnfs)
        if flow_ok:
            for start, end in zip(positions, positions[1:]):
                balance = Counter()
                for left, right in zip(path[start:end], path[start + 1:end + 1]):
                    balance[left] += 1
                    balance[right] -= 1
                expected = Counter()
                expected[path[start]] += 1
                expected[path[end]] -= 1
                if any(balance[n] != expected[n] for n in self.snapshot.satellites):
                    flow_ok = False
                    break
        add("C5", flow_ok, "flow_conservation_violation")
        cpu, bandwidth = self._demand(request, allocation)
        old_cpu, old_bandwidth = self._demand(request, credit) if credit else (Counter(), Counter())
        add("C6", known and all(demand <= self.cpu_available(n) + old_cpu[n] + EPSILON
                                for n, demand in cpu.items()), "insufficient_cpu")
        add("C7", linked and all(demand <= self.bandwidth_available(*key) + old_bandwidth[key] + EPSILON
                                 for key, demand in bandwidth.items()), "insufficient_bandwidth")
        # C8 的 0/1 決策由單一路徑與每個 VNF 的單一節點表示；不接受分數型配置。
        add("C8", all(type(n) is int for n in path + placements), "nonbinary_embedding")
        return FeasibilityReport(tuple(checks), delay)

    def _validate(self, request: SFCRequest, allocation: Allocation, credit: Allocation | None = None) -> str:
        """保留生命週期內部原介面，由完整診斷取得第一項違規原因。"""
        return self._check_allocation(request, allocation, credit).reason

    def _loads(self, cpu: Counter, bandwidth: Counter) -> dict[int, float]:
        # 公式 (4)：b_n 採相鄰鏈路保留頻寬總和；一條鏈路在兩端各計一次。
        communication = Counter()
        for (a, b), usage in bandwidth.items():
            communication[a] += usage
            communication[b] += usage
        return {n: self.cpu_weight * cpu[n] + self.bandwidth_weight * communication[n]
                for n in self.snapshot.satellites}

    def node_loads(self) -> dict[int, float]:
        """回傳每顆衛星的已保留資源負載 l_n（不將利用率誤當資源量）。"""
        return self._loads(self._cpu_used, self._bandwidth_used)

    def fairness(self) -> float:
        """以所有衛星（含零負載節點）計算 Jain 公平性。"""
        return jain_fairness(self.node_loads().values())

    def projected_loads(self, request: SFCRequest, allocation: Allocation) -> dict[int, float]:
        """只讀預覽候選配置的負載；允許 C4 超時，以支援公式 (18) 第二分支。"""
        report = self.check_allocation(request, allocation)
        if any(not c.passed and c.code != "C4" for c in report.checks):
            raise ValueError("Cannot project an invalid allocation: " + report.reason)
        cpu, bandwidth = self._demand(request, allocation)
        return self._loads(self._cpu_used + cpu, self._bandwidth_used + bandwidth)

    def _apply(self, request: SFCRequest, allocation: Allocation, sign: int):
        cpu, bandwidth = self._demand(request, allocation)
        for account, demand in ((self._cpu_used, cpu), (self._bandwidth_used, bandwidth)):
            for key, value in demand.items():
                account[key] += sign * value
                if abs(account[key]) < EPSILON:
                    del account[key]

    def reject(self, request: SFCRequest, reason: str) -> Decision:
        if request.id in self.records:
            return Decision(False, "duplicate_request_id")
        self.records[request.id] = ServiceRecord(request, ServiceStatus.REJECTED, reason=reason)
        self._event("rejected", request.id, reason)
        return Decision(False, reason)

    def admit(self, request: SFCRequest, allocation: Allocation) -> Decision:
        """通過全部約束才提交；失敗請求僅寫入拒絕事件。"""
        if request.id in self.records:
            return Decision(False, "duplicate_request_id")
        reason = self._validate(request, allocation)
        if reason:
            return self.reject(request, reason)
        self._apply(request, allocation, 1)
        self.records[request.id] = ServiceRecord(request, ServiceStatus.ACTIVE, allocation)
        self._event("accepted", request.id)
        return Decision(True, "accepted")

    def replace_allocation(self, request_id: str, allocation: Allocation) -> Decision:
        """遷移策略的掛接介面；驗證失敗保留舊配置及帳本。"""
        record = self.records.get(request_id)
        if record is None or record.status not in (ServiceStatus.ACTIVE, ServiceStatus.INTERRUPTED):
            return Decision(False, "service_not_reserved")
        reason = self._validate(record.request, allocation, credit=record.allocation)
        if reason:
            return Decision(False, reason)
        self._apply(record.request, record.allocation, -1)
        self._apply(record.request, allocation, 1)
        record.allocation = allocation
        record.status, record.reason, record.interrupted_since_s = ServiceStatus.ACTIVE, "", None
        self._event("reallocated", request_id)
        return Decision(True, "reallocated")

    def release(self, request_id: str, *, expired: bool = False) -> bool:
        """釋放一次；再次呼叫回傳 False，避免負資源帳本。"""
        record = self.records.get(request_id)
        if record is None or record.status not in (ServiceStatus.ACTIVE, ServiceStatus.INTERRUPTED):
            return False
        self._apply(record.request, record.allocation, -1)
        record.status = ServiceStatus.EXPIRED if expired else ServiceStatus.RELEASED
        record.reason = "lifetime_ended" if expired else "explicit_release"
        record.interrupted_since_s = None
        self._event(record.status.value, request_id, record.reason)
        return True

    def update(self, snapshot: NetworkSnapshot):
        """先釋放到期資源，再依新拓樸檢查其餘服務的連通性及延遲。"""
        if snapshot.time_s < self.time_s:
            raise ValueError("Simulation time cannot move backwards")
        if set(snapshot.satellites) != set(self.snapshot.satellites):
            raise ValueError("Changing satellite IDs requires a new ResourceManager")
        # Hardware capacities are fixed in this environment; only topology/delay changes.
        if any(snapshot.satellites[n].cpu_capacity != old.cpu_capacity
               for n, old in self.snapshot.satellites.items()):
            raise ValueError("Changing satellite CPU capacity is unsupported")
        if any(link.capacity_mhz + EPSILON < self._bandwidth_used[key]
               for key, link in snapshot.links.items()):
            raise ValueError("New link capacity cannot be below existing reservations")
        self.snapshot = snapshot
        for request_id, record in self.records.items():
            if record.status not in (ServiceStatus.ACTIVE, ServiceStatus.INTERRUPTED):
                continue
            if record.request.expires_s <= self.time_s:
                self.release(request_id, expired=True)
                continue
            edges = [edge_key(a, b) for a, b in zip(record.allocation.path, record.allocation.path[1:])]
            reason = ""
            if any(key not in snapshot.links for key in edges):
                reason = "link_unavailable"
            elif sum(snapshot.links[key].delay_s for key in edges) > record.request.max_delay_s + 1e-12:
                reason = "delay_exceeded"
            if reason:
                if record.status != ServiceStatus.INTERRUPTED:
                    record.interruption_count += 1
                    record.interrupted_since_s = self.time_s
                    self._event("interrupted", request_id, reason)
                record.status, record.reason = ServiceStatus.INTERRUPTED, reason
            elif record.status == ServiceStatus.INTERRUPTED:
                record.status, record.reason, record.interrupted_since_s = ServiceStatus.ACTIVE, "", None
                self._event("restored", request_id)

    def pending_migrations(self) -> list[str]:
        return [key for key, record in self.records.items() if record.status == ServiceStatus.INTERRUPTED]

    def metrics(self) -> dict:
        counts = Counter(record.status.value for record in self.records.values())
        # 指標與 MDP 共用同一套公式，避免公平性定義分歧。
        loads = self.node_loads()
        fairness = jain_fairness(loads.values())
        accepted = sum(record.allocation is not None for record in self.records.values())
        total = len(self.records)
        return {
            "time_s": self.time_s, "satellites": len(loads), "links": len(self.snapshot.links),
            "requests": total, "accepted": accepted,
            "admission_ratio": accepted / total if total else 0.0,
            "statuses": {status.value: counts[status.value] for status in ServiceStatus},
            "cpu_reserved": sum(self._cpu_used.values()),
            "bandwidth_reserved_mhz": sum(self._bandwidth_used.values()),
            "load_fairness": fairness,
            "node_loads": loads,
            "interruption_events": sum(r.interruption_count for r in self.records.values()),
        }

    def assert_invariants(self):
        """從存活服務重新推算帳本，驗證不漏扣、不超額、不重複釋放。"""
        cpu, bandwidth = Counter(), Counter()
        for record in self.records.values():
            if record.status in (ServiceStatus.ACTIVE, ServiceStatus.INTERRUPTED):
                c, b = self._demand(record.request, record.allocation)
                cpu.update(c)
                bandwidth.update(b)
        for expected, actual in ((cpu, self._cpu_used), (bandwidth, self._bandwidth_used)):
            for key in set(expected) | set(actual):
                assert abs(expected[key] - actual[key]) < EPSILON, ("ledger_mismatch", key)
                assert actual[key] >= -EPSILON, ("negative_reservation", key)
        for node in self.snapshot.satellites:
            assert self.cpu_available(node) >= -EPSILON, ("cpu_overallocated", node)
        for edge in self.snapshot.links:
            assert self.bandwidth_available(*edge) >= -EPSILON, ("bandwidth_overallocated", edge)
