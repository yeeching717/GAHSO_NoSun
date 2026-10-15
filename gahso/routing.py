"""Read-only delay bounds and admission diagnostics; never selects Actor actions."""
from heapq import heappop, heappush
from math import inf, isfinite
from .resources import EPSILON


def remaining_delays(manager, destination, bandwidth=0., excluded=()):
    blocked = set(excluded)
    if destination in blocked:
        return {}
    distances, queue = {destination: 0.}, [(0., destination)]
    while queue:
        cost, node = heappop(queue)
        if cost != distances[node]:
            continue
        for neighbor in manager.snapshot.neighbors(node):
            if neighbor in blocked or manager.bandwidth_available(node, neighbor) + EPSILON < bandwidth:
                continue
            edge = (min(node, neighbor), max(node, neighbor))
            candidate = cost + manager.snapshot.links[edge].delay_s
            if candidate < distances.get(neighbor, inf):
                distances[neighbor] = candidate
                heappush(queue, (candidate, neighbor))
    return distances


def arrival_diagnostics(manager, request, endpoint_placement_mode="fixed"):
    physical = remaining_delays(manager, request.destination).get(request.source, inf)
    available = remaining_delays(manager, request.destination, request.bandwidth_mhz).get(request.source, inf)
    if not isfinite(physical):
        status = "initial_no_physical_route"
    elif physical > request.max_delay_s + 1e-12:
        status = "initial_delay_infeasible"
    elif not isfinite(available):
        status = "initial_bandwidth_infeasible"
    elif available > request.max_delay_s + 1e-12:
        status = "initial_bandwidth_delay_infeasible"
    else:
        status = "initial_route_feasible"
    if endpoint_placement_mode not in ("fixed", "neighbor_allowed"):
        raise ValueError("Unknown endpoint placement mode")
    source_candidates = [request.source]
    destination_candidates = [request.destination]
    if endpoint_placement_mode == "neighbor_allowed":
        source_candidates += list(manager.snapshot.neighbors(request.source))
        destination_candidates += list(manager.snapshot.neighbors(request.destination))
    source_cpu = {str(n): manager.cpu_available(n) for n in source_candidates}
    destination_cpu = {str(n): manager.cpu_available(n) for n in destination_candidates}
    if request.source == request.destination and endpoint_placement_mode == "fixed":
        endpoint_ok = manager.cpu_available(request.source) + EPSILON >= (
            request.vnfs[0].cpu + request.vnfs[-1].cpu)
    else:
        endpoint_ok = (any(v + EPSILON >= request.vnfs[0].cpu for v in source_cpu.values()) and
                       any(v + EPSILON >= request.vnfs[-1].cpu for v in destination_cpu.values()))
    return {"route_status": status,
            "physical_min_delay_s": physical if isfinite(physical) else None,
            "bandwidth_min_delay_s": available if isfinite(available) else None,
            "endpoint_cpu_feasible": endpoint_ok,
            "endpoint_placement_mode": endpoint_placement_mode,
            "source_endpoint_cpu_demand": request.vnfs[0].cpu,
            "destination_endpoint_cpu_demand": request.vnfs[-1].cpu,
            "source_endpoint_available_cpu": source_cpu,
            "destination_endpoint_available_cpu": destination_cpu,
            "route_feasible": status == "initial_route_feasible",
            "note": "Route feasibility does not prove full VNF placement feasibility."}


def failure_category(initial, reason, accepted=False):
    if accepted:
        return "accepted"
    if not initial["route_feasible"]:
        return initial["route_status"]
    if not initial["endpoint_cpu_feasible"]:
        return "initial_endpoint_cpu_infeasible"
    if reason in ("vnf_embedding_failed", "insufficient_cpu"):
        return "vnf_placement_failure"
    if reason in ("delay_exceeded", "delay_lower_bound_exceeded"):
        return "policy_delay_failure"
    return "policy_route_or_placement_failure"
