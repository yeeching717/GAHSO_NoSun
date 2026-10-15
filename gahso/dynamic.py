"""事件驅動部署：軌道變動、可重現斷鏈、修復及明確分母的持續率。"""
from dataclasses import asdict
from math import floor, isfinite
from random import Random
from time import perf_counter
from .network import LEOConstellation, NetworkSnapshot
from .resources import ResourceManager, ServiceStatus
from .migration import MigrationPlanner
from .routing import arrival_diagnostics, failure_category


class FaultTopology:
    """每個 slot 獨立以 p 移除鏈路，故障維持至該 slot 結束；同種子跨策略相同。"""
    def __init__(self, constellation, *, probability=0.0, slot_s=10.0, seed=123):
        if not isfinite(probability) or not 0 <= probability <= 1 or not isfinite(slot_s) or slot_s <= 0:
            raise ValueError("Invalid fault probability or slot duration")
        self.constellation, self.probability, self.slot_s, self.seed = constellation, probability, slot_s, seed

    def snapshot(self, time_s):
        base = self.constellation.snapshot(time_s)
        rng = Random(self.seed + floor(time_s / self.slot_s) * 1000003)
        # 對固定的全部無向 node pair 抽樣，拓樸新增／移除不會改變其他邊的故障命運。
        ids = sorted(base.satellites)
        disabled = { (a, b) for i, a in enumerate(ids) for b in ids[i+1:]
                     if rng.random() < self.probability }
        return NetworkSnapshot(time_s, base.satellites,
                               {e: link for e, link in base.links.items() if e not in disabled}, base.nominal_links)


def validate_active(manager):
    """重算帳本，並針對每個 ACTIVE 服務檢查全部 C1–C8。"""
    manager.assert_invariants()
    for key, record in manager.records.items():
        if record.status == ServiceStatus.ACTIVE:
            view = manager.planning_copy(key)
            report = view.check_allocation(record.request, record.allocation)
            if not report.feasible:
                raise AssertionError((key, report.violations))


def run_dynamic(config, requests, policy, *, topology=None, repair=True, migration_config=None):
    """策略回傳候選 Allocation；只有本 runner 對正式帳本 admit。

    執行時間分開記錄部署決策及修復。持續率是事件後／服務時間模型，
    不包含 checkpoint 載入、訓練或真實 live migration 停機時間。
    """
    requests = sorted(requests, key=lambda q: (q.arrival_s, q.id))
    if len({q.id for q in requests}) != len(requests):
        raise ValueError("Duplicate workload request IDs")
    if any(not 0 <= q.arrival_s < config.horizon_s for q in requests):
        raise ValueError("Requests must arrive within the horizon")
    topology = topology or FaultTopology(LEOConstellation(config.constellation),
                                         slot_s=config.topology_step_s)
    manager = ResourceManager(topology.snapshot(0), config.cpu_load_weight, config.bandwidth_load_weight)
    planner = MigrationPlanner(migration_config)
    end = max([config.horizon_s] + [q.expires_s for q in requests]) if config.drain else config.horizon_s
    ticks = {i * config.topology_step_s for i in range(int(end / config.topology_step_s) + 1)}
    times = sorted(ticks | {end, config.horizon_s} | {q.arrival_s for q in requests} |
                   {q.expires_s for q in requests if q.expires_s <= end})
    arrivals = {}
    for q in requests:
        arrivals.setdefault(q.arrival_s, []).append(q)
    migration_events, timeline, decisions = [], [], []
    failed_after_repair = set()
    affected_services = set()
    available_time = reserved_time = decision_seconds = repair_seconds = 0.0
    active_samples = alive_samples = raw_active_samples = 0
    previous = 0.0
    for time_s in times:
        alive = [r for r in manager.records.values() if r.status in (ServiceStatus.ACTIVE, ServiceStatus.INTERRUPTED)]
        reserved_time += (time_s - previous) * len(alive)
        available_time += (time_s - previous) * sum(r.status == ServiceStatus.ACTIVE for r in alive)
        previous = time_s
        manager.update(topology.snapshot(time_s))
        alive = [r for r in manager.records.values() if r.status in (ServiceStatus.ACTIVE, ServiceStatus.INTERRUPTED)]
        raw_active = sum(r.status == ServiceStatus.ACTIVE for r in alive)
        affected_services.update(manager.pending_migrations())
        start = perf_counter()
        if repair:
            for key in sorted(manager.pending_migrations()):
                migration_events.append(planner.repair(manager, key))
        repair_seconds += perf_counter() - start
        remaining = manager.pending_migrations()
        failed_after_repair.update(remaining)
        if time_s in ticks:
            alive_samples += len(alive)
            raw_active_samples += raw_active
            active_samples += sum(r.status == ServiceStatus.ACTIVE for r in alive)
        for q in arrivals.get(time_s, []):
            initial = arrival_diagnostics(manager, q, config.mdp.endpoint_placement_mode)
            start = perf_counter()
            allocation = policy(manager, q)
            decision = manager.admit(q, allocation) if allocation else manager.reject(q, "policy_no_feasible_allocation")
            elapsed = perf_counter() - start
            decision_seconds += elapsed
            info = (getattr(policy, "last_report", None) or {}).get("info", {})
            reason = info.get("reason", decision.reason)
            decisions.append({"request_id": q.id, "accepted": decision.accepted, "seconds": elapsed,
                              "reason": reason, "initial_diagnostics": initial,
                              "failure_category": failure_category(initial, reason, decision.accepted),
                              "fairness": manager.fairness(),
                              "allocation": asdict(allocation) if decision.accepted else None})
        validate_active(manager)
        timeline.append({"time_s": time_s, "active": manager.metrics()["statuses"]["active"],
                         "interrupted": len(remaining), "fairness": manager.fairness(),
                         "links": len(manager.snapshot.links)})
    final = manager.metrics()
    accepted = final["accepted"]
    batches = [sum(d["seconds"] for d in decisions[i:i+100])
               for i in range(0, len(decisions) - 99, 100)]
    return {"requests": len(requests), "accepted": accepted, "acceptance_rate": final["admission_ratio"],
            "mean_arrival_jain": sum(d["fairness"] for d in decisions) / len(decisions) if decisions else None,
            "seconds_per_100_sfc": decision_seconds * 100 / len(requests) if requests else None,
            "complete_100_sfc_batches_seconds": batches,
            "decision_seconds": decision_seconds, "repair_seconds": repair_seconds,
            "service_time_availability": available_time / reserved_time if reserved_time else None,
            "available_service_seconds": available_time, "reserved_service_seconds": reserved_time,
            "event_continuity_before_repair": raw_active_samples / alive_samples if alive_samples else None,
            "event_continuity_after_repair": active_samples / alive_samples if alive_samples else None,
            "event_alive_denominator": alive_samples,
            "uninterrupted_after_repair_rate": (accepted - len(failed_after_repair)) / accepted if accepted else None,
            "repair_attempts": len(migration_events),
            "affected_services": len(affected_services),
            "repair_success_rate": sum(e["repaired"] for e in migration_events) / len(migration_events) if migration_events else None,
            "vnf_migrations_per_accepted_sfc": sum(e["vnf_migrations"] for e in migration_events) / accepted if accepted else None,
            "vnf_migrations_per_affected_sfc": sum(e["vnf_migrations"] for e in migration_events) / len(affected_services) if affected_services else None,
            "repair_successes": sum(e["repaired"] for e in migration_events),
            "vnf_migrations": sum(e["vnf_migrations"] for e in migration_events),
            "reroutes": sum(e["rerouted"] for e in migration_events),
            "limited_searches": sum(not e["search_exhausted"] for e in migration_events),
            "environment_checks_passed": True, "final": final, "timeline": timeline,
            "decisions": decisions, "migration_events": migration_events,
            "events": manager.events}
