"""Algorithm 2 的有序貪婪放置；純預覽，永遠不修改資源帳本。"""
from dataclasses import dataclass
from .resources import Allocation, ResourceManager, EPSILON
from .services import SFCRequest


@dataclass(frozen=True)
class EmbeddingResult:
    path: tuple[int, ...]
    placements: tuple[int | None, ...]
    candidate_counts: tuple[int, ...] = ()
    failure_reasons: tuple[str | None, ...] = ()
    endpoint_placement_mode: str = "fixed"

    @property
    def unembedded(self) -> int:
        """公式 (18) 的 n_embed：尚未能放置的 VNF 數量。"""
        return sum(node is None for node in self.placements)

    @property
    def allocation(self) -> Allocation | None:
        if self.unembedded:
            return None
        return Allocation(self.path, self.placements, self.endpoint_placement_mode)


def embed_vnfs(manager: ResourceManager, request: SFCRequest,
               path: tuple[int, ...]) -> EmbeddingResult:
    """在完整或部分候選路徑上放置 VNF，遵守 C1/C2 端點及有序處理。

    先暫扣首尾 VNF，再從不倒退的路徑範圍選剩餘 CPU 最大的節點。
    尾端尚未到達時，最後一個 VNF 保留 None，供失敗獎勵計算。
    論文簡略 Algorithm 2 未列出所有細節；這裡明確補上端點與順序限制。
    """
    path = tuple(path)
    if not path or path[0] != request.source or len(set(path)) != len(path):
        raise ValueError("Embedding needs a simple path starting at the source")
    if any(n not in manager.snapshot.satellites for n in path):
        raise ValueError("Unknown node in embedding path")
    available = {node: manager.cpu_available(node) for node in path}
    placements = [None] * len(request.vnfs)

    def reserve(index, node):
        if available[node] + EPSILON >= request.vnfs[index].cpu:
            available[node] -= request.vnfs[index].cpu
            placements[index] = node

    reserve(0, request.source)
    if path[-1] == request.destination:
        reserve(len(request.vnfs) - 1, request.destination)
    first_index = 0
    for i, vnf in enumerate(request.vnfs[1:-1], start=1):
        choices = [j for j in range(first_index, len(path)) if available[path[j]] + EPSILON >= vnf.cpu]
        if not choices:
            # 後續 VNF 不越過尚未放置的前序 VNF。
            break
        first_index = max(choices, key=lambda j: (available[path[j]], -j))
        node = path[first_index]
        available[node] -= vnf.cpu
        placements[i] = node
    return EmbeddingResult(path, tuple(placements))


def embed_vnfs(manager: ResourceManager, request: SFCRequest, path: tuple[int, ...],
               endpoint_placement_mode: str = "fixed") -> EmbeddingResult:
    """Ordered placement with endpoint-mode and per-VNF failure diagnostics."""
    path = tuple(path)
    if not path or path[0] != request.source or len(set(path)) != len(path):
        raise ValueError("Embedding needs a simple path starting at the source")
    if any(n not in manager.snapshot.satellites for n in path):
        raise ValueError("Unknown node in embedding path")
    if endpoint_placement_mode not in ("fixed", "neighbor_allowed"):
        raise ValueError("Unknown endpoint placement mode")
    available = {node: manager.cpu_available(node) for node in path}
    placements = [None] * len(request.vnfs)
    candidate_counts = [0] * len(request.vnfs)
    failures = [None] * len(request.vnfs)
    source_indices = [0]
    destination_indices = [len(path) - 1] if path[-1] == request.destination else []
    if endpoint_placement_mode == "neighbor_allowed":
        if len(path) > 1 and path[1] in manager.snapshot.neighbors(request.source):
            source_indices.append(1)
        if (len(path) > 1 and path[-1] == request.destination and
                path[-2] in manager.snapshot.neighbors(request.destination)):
            destination_indices.append(len(path) - 2)

    def reserve_endpoint(index, indices, lower=0):
        choices = [j for j in dict.fromkeys(indices) if j >= lower and
                   available[path[j]] + EPSILON >= request.vnfs[index].cpu]
        candidate_counts[index] = len(choices)
        if not choices:
            failures[index] = "insufficient_cpu_at_allowed_endpoint"
            return None
        chosen = max(choices, key=lambda j: (available[path[j]], -j))
        available[path[chosen]] -= request.vnfs[index].cpu
        placements[index] = path[chosen]
        return chosen

    first_index = reserve_endpoint(0, source_indices)
    last_index = reserve_endpoint(len(request.vnfs) - 1, destination_indices,
                                  0 if first_index is None else first_index)
    lower = 0 if first_index is None else first_index
    upper = len(path) - 1 if last_index is None else last_index
    for i, vnf in enumerate(request.vnfs[1:-1], start=1):
        choices = [j for j in range(lower, upper + 1)
                   if available[path[j]] + EPSILON >= vnf.cpu]
        candidate_counts[i] = len(choices)
        if not choices:
            failures[i] = "no_ordered_cpu_candidate"
            break
        lower = max(choices, key=lambda j: (available[path[j]], -j))
        available[path[lower]] -= vnf.cpu
        placements[i] = path[lower]
    for i, node in enumerate(placements):
        if node is None and failures[i] is None:
            failures[i] = "not_attempted_after_previous_failure"
    return EmbeddingResult(path, tuple(placements), tuple(candidate_counts), tuple(failures),
                           endpoint_placement_mode)
