"""P2 候選搜索與原子遷移；CPU 上逐服務搜索，明確標示搜索上限。"""
from dataclasses import asdict, dataclass
from heapq import heappop, heappush
from itertools import combinations_with_replacement
from math import isfinite
from .network import edge_key
from .resources import Allocation, ServiceStatus, jain_fairness


@dataclass(frozen=True)
class MigrationConfig:
    migration_weight: float = 1.0
    fairness_weight: float = 1.0
    max_paths: int = 64
    max_candidates: int = 4096
    max_expansions: int = 10000

    def __post_init__(self):
        if any(not isfinite(v) or v < 0 for v in (self.migration_weight, self.fairness_weight)):
            raise ValueError("Migration weights must be finite and nonnegative")
        if any(type(v) is not int or v <= 0 for v in
               (self.max_paths, self.max_candidates, self.max_expansions)):
            raise ValueError("Search limits must be positive integers")


class MigrationPlanner:
    """對可行解最小化 gamma2*遷移數 - 1 - gamma1*全網 Jain。

    先保證服務持續，再比較式 (15) 的可行解；失敗保留舊配置以待下次修復。
    逐服務 greedy，不宣稱跨服務聯合最優；未觸上限時窮舉所有可行簡單路徑配置。
    """
    def __init__(self, config=None):
        self.config = config or MigrationConfig()

    def repair(self, manager, request_id):
        record = manager.records.get(request_id)
        if record is None or record.status != ServiceStatus.INTERRUPTED:
            raise ValueError("Migration requires an interrupted service")
        old, q, cfg = record.allocation, record.request, self.config
        view = manager.planning_copy(request_id)
        queue = [(0.0, (q.source,))]
        paths = candidates = expansions = 0
        limited = False
        best = None
        # 最短延遲優先展開；頻寬與延遲剪枝不排除任何可行解。
        while queue:
            if paths >= cfg.max_paths or candidates >= cfg.max_candidates or expansions >= cfg.max_expansions:
                limited = True
                break
            delay, path = heappop(queue)
            expansions += 1
            if path[-1] == q.destination:
                paths += 1
                for middle in combinations_with_replacement(range(len(path)), len(q.vnfs) - 2):
                    if candidates >= cfg.max_candidates:
                        limited = True
                        break
                    candidates += 1
                    placement = (q.source,) + tuple(path[i] for i in middle) + (q.destination,)
                    allocation = Allocation(path, placement)
                    if not view.check_allocation(q, allocation).feasible:
                        continue
                    moved = sum(a != b for a, b in zip(old.vnf_nodes, placement))
                    fairness = jain_fairness(view.projected_loads(q, allocation).values())
                    cost = cfg.migration_weight * moved - 1 - cfg.fairness_weight * fairness
                    key = (cost, moved, delay, path, placement)
                    if best is None or key < best[0]:
                        best = (key, allocation, fairness, moved)
                if limited:
                    break
            else:
                for neighbor in view.snapshot.neighbors(path[-1]):
                    if neighbor in path or view.bandwidth_available(path[-1], neighbor) + 1e-8 < q.bandwidth_mhz:
                        continue
                    next_delay = delay + view.snapshot.links[edge_key(path[-1], neighbor)].delay_s
                    if next_delay <= q.max_delay_s + 1e-12:
                        heappush(queue, (next_delay, path + (neighbor,)))
        result = {"request_id": request_id, "time_s": manager.time_s,
                  "repaired": False, "vnf_migrations": 0, "rerouted": False,
                  "paths_examined": paths, "candidates_examined": candidates,
                  "expansions": expansions, "search_exhausted": not limited,
                  "old_allocation": asdict(old), "new_allocation": None,
                  "reason": "search_limit" if limited else "no_feasible_allocation"}
        if best is not None:
            _, allocation, fairness, moved = best
            decision = manager.replace_allocation(request_id, allocation)
            result.update(repaired=decision.accepted, reason=decision.reason)
            if decision.accepted:
                result.update(vnf_migrations=moved, rerouted=old.path != allocation.path,
                              new_allocation=asdict(allocation), objective=best[0][0],
                              fairness=fairness)
        manager.assert_invariants()
        return result
