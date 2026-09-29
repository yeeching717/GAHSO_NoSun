"""事件驅動的背景服務模擬；決策策略產生候選，ResourceManager 統一提交。"""
from dataclasses import asdict, dataclass, field
from heapq import heappop, heappush
from math import isfinite
from typing import Callable
from .baseline import shortest_delay_greedy
from .mdp import MDPConfig
from .gat_config import GATConfig, GATFeatureConfig
from .agent_config import AgentConfig
from .network import ConstellationConfig, LEOConstellation
from .resources import Allocation, ResourceManager
from .services import RequestConfig, RequestGenerator, SFCRequest


@dataclass(frozen=True)
class SimulationConfig:
    horizon_s: float = 600.0
    topology_step_s: float = 10.0
    drain: bool = True
    cpu_load_weight: float = 1.0
    bandwidth_load_weight: float = 1.0
    constellation: ConstellationConfig = field(default_factory=ConstellationConfig)
    requests: RequestConfig = field(default_factory=RequestConfig)
    mdp: MDPConfig = field(default_factory=MDPConfig)
    gat_features: GATFeatureConfig = field(default_factory=GATFeatureConfig)
    gat: GATConfig = field(default_factory=GATConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)

    def __post_init__(self):
        if not all(isfinite(x) and x > 0 for x in (self.horizon_s, self.topology_step_s)):
            raise ValueError("Horizon and topology step must be finite and positive")
        if not all(isfinite(x) and x >= 0 for x in (self.cpu_load_weight, self.bandwidth_load_weight)):
            raise ValueError("Load weights must be finite and nonnegative")
        if self.cpu_load_weight + self.bandwidth_load_weight <= 0:
            raise ValueError("At least one load weight must be positive")

    @classmethod
    def from_dict(cls, data: dict):
        payload = dict(data)
        payload["constellation"] = ConstellationConfig(**payload.get("constellation", {}))
        payload["requests"] = RequestConfig(**payload.get("requests", {}))
        payload["mdp"] = MDPConfig.from_dict(payload.get("mdp", {}))
        payload["gat_features"] = GATFeatureConfig(**payload.get("gat_features", {}))
        payload["gat"] = GATConfig(**payload.get("gat", {}))
        payload["agent"] = AgentConfig(**payload.get("agent", {}))
        return cls(**payload)


Policy = Callable[[ResourceManager, SFCRequest], Allocation | None]


class Simulation:
    """處理到達、到期與拓樸事件；每次 run 使用獨立帳本。"""
    def __init__(self, config: SimulationConfig | None = None, policy: Policy = shortest_delay_greedy):
        self.config = config or SimulationConfig()
        self.constellation = LEOConstellation(self.config.constellation)
        self.resources = ResourceManager(self.constellation.snapshot(0),
                                         self.config.cpu_load_weight, self.config.bandwidth_load_weight)
        self.policy = policy
        self.timeline = []
        self._ran = False

    def run(self, requests: list[SFCRequest] | None = None) -> dict:
        if self._ran:
            raise RuntimeError("Create a new Simulation for each run")
        cfg = self.config
        if requests is None:
            requests = RequestGenerator(cfg.requests).generate(cfg.horizon_s, cfg.constellation.satellites)
        requests = sorted(requests, key=lambda request: (request.arrival_s, request.id))
        if len({request.id for request in requests}) != len(requests):
            raise ValueError("Workload contains duplicate request IDs")
        if any(not 0 <= request.arrival_s < cfg.horizon_s for request in requests):
            raise ValueError("Requests must arrive within [0, horizon_s)")
        self._ran = True
        queue = []
        # 同時事件優先順序：到期→拓樸更新→新請求；先歸還資源再接新流量。
        for index, request in enumerate(requests):
            heappush(queue, (request.arrival_s, 2, index, request))
        # drain 開啟時，觀察期間結束後繼續處理直到所有請求到期。
        end_s = max([cfg.horizon_s] + [q.expires_s for q in requests]) if cfg.drain else cfg.horizon_s
        tick = 0
        while tick * cfg.topology_step_s < end_s:
            heappush(queue, (tick * cfg.topology_step_s, 1, tick, None))
            tick += 1
        heappush(queue, (cfg.horizon_s, 1, tick, None))
        heappush(queue, (end_s, 1, tick + 1, None))
        horizon_metrics = None
        while queue:
            time_s, priority, index, request = heappop(queue)
            if time_s > end_s:
                break
            self.resources.update(self.constellation.snapshot(time_s))
            if priority == 2:
                allocation = self.policy(self.resources, request)
                if allocation is None:
                    decision = self.resources.reject(request, "baseline_no_feasible_allocation")
                else:
                    decision = self.resources.admit(request, allocation)
                if decision.accepted and request.expires_s <= end_s:
                    heappush(queue, (request.expires_s, 0, index, None))
            # 每個事件後從服務紀錄重新驗算帳本，立即捕捉資源洩漏。
            self.resources.assert_invariants()
            metrics = self.resources.metrics()
            if self.timeline and self.timeline[-1]["time_s"] == time_s:
                self.timeline[-1] = metrics
            else:
                self.timeline.append(metrics)
            if time_s == cfg.horizon_s:
                horizon_metrics = metrics
        return {
            "policy": getattr(self.policy, "__name__", self.policy.__class__.__name__),
            "config": asdict(cfg),
            "at_horizon": horizon_metrics,
            "final": self.resources.metrics(),
            "generated_requests": len(requests),
            "orbital_period_s": self.constellation.orbital_period_s,
            "events": list(self.resources.events),
            "timeline": self.timeline,
            "services": [
                {"request": asdict(record.request), "status": record.status.value,
                 "allocation": asdict(record.allocation) if record.allocation else None,
                 "reason": record.reason, "interruption_count": record.interruption_count}
                for record in self.resources.records.values()
            ],
        }
