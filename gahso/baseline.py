"""基準示範：Dijkstra 選路＋Algorithm 2 有序放置；尚非學習式 GAHSO。"""
from heapq import heappop, heappush
from .resources import Allocation, ResourceManager
from .services import SFCRequest
from .embedding import embed_vnfs


def shortest_delay_greedy(manager: ResourceManager, request: SFCRequest) -> Allocation | None:
    """先找頻寬足夠的最短延遲路徑，再呼叫與 MDP 共用的 VNF 放置器。"""
    if request.source not in manager.snapshot.satellites or request.destination not in manager.snapshot.satellites:
        return None
    # previous 保存前驅節點，供到達目的地後重建整條路徑。
    distances, previous = {request.source: 0.0}, {}
    queue = [(0.0, request.source)]
    while queue:
        cost, node = heappop(queue)
        if cost != distances[node]:
            continue
        if node == request.destination:
            break
        # 移除頻寬不足的邊；邊權重只採傳播延遲。
        for neighbor in manager.snapshot.neighbors(node):
            if manager.bandwidth_available(node, neighbor) + 1e-8 < request.bandwidth_mhz:
                continue
            link = manager.snapshot.links[tuple(sorted((node, neighbor)))]
            next_cost = cost + link.delay_s
            if next_cost < distances.get(neighbor, float("inf")):
                distances[neighbor], previous[neighbor] = next_cost, node
                heappush(queue, (next_cost, neighbor))
    if request.destination not in distances or distances[request.destination] > request.max_delay_s:
        return None
    path = [request.destination]
    while path[-1] != request.source:
        path.append(previous[path[-1]])
    path.reverse()
    # 放置器只使用暫存 CPU；正式配置由 ResourceManager.admit() 執行。
    return embed_vnfs(manager, request, tuple(path)).allocation
