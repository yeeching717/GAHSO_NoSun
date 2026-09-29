"""將 GAHSO MDP observation 編碼為可微分模型的輸入；本模組需要 PyTorch。"""
from dataclasses import dataclass, replace
from math import isfinite
from typing import Mapping
import torch
from torch import Tensor
from .gat_config import GATFeatureConfig
from .mdp import SFCOrchestrationEnv


@dataclass(frozen=True)
class GATInputs:
    """單張圖；edge_index[0] 是訊息來源 j，[1] 是接收者 i。

    x: [N,F_node]；edge_attr: [M,2]；edge_index: [2,M]。
    M=2|E|，因 MDP 已將每條無向 ISL 展開為兩個訊息方向。
    position 與 action_mask 另保留，供後續 Actor/Critic 使用。
    """
    x: Tensor
    edge_index: Tensor
    edge_attr: Tensor
    position: Tensor
    action_mask: Tensor
    feasible_action_mask: Tensor
    request_features: Tensor
    node_ids: tuple[int, ...]
    node_feature_names: tuple[str, ...]
    edge_feature_names: tuple[str, ...]
    request_feature_names: tuple[str, ...]

    def to(self, device=None, *, dtype=None) -> "GATInputs":
        """搬移所有張量；索引仍為 long、遮罩仍為 bool，不隨浮點 dtype 改變。"""
        dtype = self.x.dtype if dtype is None else dtype
        if dtype not in (torch.float32, torch.float64):
            raise ValueError("GAT inputs support float32 or float64")
        return replace(
            self, x=self.x.to(device=device, dtype=dtype),
            edge_index=self.edge_index.to(device=device),
            edge_attr=self.edge_attr.to(device=device, dtype=dtype),
            position=self.position.to(device=device, dtype=dtype),
            action_mask=self.action_mask.to(device=device),
            feasible_action_mask=self.feasible_action_mask.to(device=device),
            request_features=self.request_features.to(device=device, dtype=dtype),
        )


class GATFeatureBuilder:
    """保留有序 VNF 需求，而非只使用服務類型；所有縮放常數固定以免樣本間漂移。"""
    BASE_NODE_NAMES = SFCOrchestrationEnv.NODE_FEATURE_NAMES
    EDGE_NAMES = SFCOrchestrationEnv.EDGE_FEATURE_NAMES

    def __init__(self, config: GATFeatureConfig | None = None):
        self.config = config or GATFeatureConfig()

    @property
    def request_feature_names(self):
        return (
            "bandwidth_demand_scaled", "delay_limit_scaled", "vnf_count_scaled",
            "remaining_delay_scaled", "steps_remaining_ratio",
        ) + tuple(f"vnf_cpu_{i}" for i in range(self.config.max_vnfs)) + tuple(
            f"vnf_present_{i}" for i in range(self.config.max_vnfs)
        )

    @property
    def node_feature_names(self):
        return self.BASE_NODE_NAMES + ("cpu_remaining_scaled", "path_order_scaled",) + self.request_feature_names

    def build(self, observation: Mapping, *, device=None, dtype=torch.float32) -> GATInputs:
        """建立獨立 tensor，不修改 MDP observation 或正式資源帳本。"""
        if dtype not in (torch.float32, torch.float64):
            raise ValueError("GAT inputs support float32 or float64")
        cfg = self.config
        ids = tuple(observation["node_ids"])
        if not ids or any(type(n) is not int for n in ids) or len(set(ids)) != len(ids):
            raise ValueError("node_ids must contain unique integer IDs")
        if tuple(observation["node_feature_names"]) != self.BASE_NODE_NAMES:
            raise ValueError("Unexpected MDP node feature schema")
        if tuple(observation["edge_feature_names"]) != self.EDGE_NAMES:
            raise ValueError("Unexpected MDP edge feature schema")
        nodes = tuple(observation["node_features"])
        if len(nodes) != len(ids) or any(len(row) != len(self.BASE_NODE_NAMES) for row in nodes):
            raise ValueError("Node feature shape does not match node_ids/schema")
        edges = tuple(observation["edge_index"])
        attrs = tuple(observation["edge_features"])
        if len(edges) != 2 or len(edges[0]) != len(edges[1]) or len(attrs) != len(edges[0]):
            raise ValueError("edge_index and edge_features must align")
        if any(type(i) is not int or not 0 <= i < len(ids) for row in edges for i in row):
            raise ValueError("Edge endpoints must be valid node row indices")
        if len(set(zip(*edges))) != len(attrs):
            raise ValueError("Duplicate directed edges")
        if any(len(row) != cfg.edge_dim for row in attrs):
            raise ValueError("Each edge must have bandwidth and delay features")
        for name in ("action_mask", "feasible_action_mask"):
            mask = observation[name]
            if len(mask) != len(ids) or any(type(v) is not bool for v in mask):
                raise ValueError(name + " must be a boolean vector aligned with node_ids")
        position = tuple(observation["position"])
        if len(position) != len(ids) or any(v not in (0, 1) for v in position):
            raise ValueError("Position must be a one-hot vector")
        path = tuple(observation["path"])
        if len(set(path)) != len(path) or any(n not in ids for n in path):
            raise ValueError("Path must contain distinct known node IDs")
        expected_position = tuple(int(bool(path) and n == path[-1]) for n in ids)
        if position != expected_position:
            raise ValueError("Position and path are inconsistent")
        max_steps = observation["max_steps"]
        remaining = observation["steps_remaining"]
        step_count = observation["step_count"]
        if (type(max_steps) is not int or max_steps < 1 or type(step_count) is not int
                or step_count < 0 or remaining != max(0, max_steps - step_count)):
            raise ValueError("Invalid decision-step counters")
        delay = observation["accumulated_delay_s"]
        if not isfinite(delay) or delay < 0:
            raise ValueError("Accumulated delay must be finite and nonnegative")

        request = observation["request"]
        if request is None:
            if path:
                raise ValueError("An idle observation cannot have a request path")
            context = [0.0] * cfg.request_dim
        else:
            if request["source"] not in ids or request["destination"] not in ids:
                raise ValueError("Unknown request endpoint")
            if not path or path[0] != request["source"]:
                raise ValueError("Request path must start at its source")
            vnfs = request["vnfs"]
            if not 2 <= len(vnfs) <= cfg.max_vnfs:
                raise ValueError(f"VNF count must be in [2, {cfg.max_vnfs}]; increase max_vnfs instead of truncating")
            demands = [v["cpu"] for v in vnfs]
            if any(not isfinite(v) or v <= 0 for v in demands):
                raise ValueError("VNF CPU demands must be finite and positive")
            if any(not isfinite(request[k]) or request[k] <= 0 for k in ("bandwidth_mhz", "max_delay_s")):
                raise ValueError("Invalid service bandwidth or delay requirement")
            # 保留超時後的負值，讓模型可以辨識已經違反的延遲限制。
            context = [
                request["bandwidth_mhz"] / cfg.bandwidth_demand_scale_mhz,
                request["max_delay_s"] / cfg.delay_scale_s,
                len(vnfs) / cfg.max_vnfs,
                (request["max_delay_s"] - delay) / cfg.delay_scale_s,
                remaining / max_steps,
            ]
            context += [v / cfg.cpu_demand_scale for v in demands] + [0.0] * (cfg.max_vnfs - len(vnfs))
            context += [1.0] * len(vnfs) + [0.0] * (cfg.max_vnfs - len(vnfs))
        # 利用率相同的 400/600 CPU 衛星仍須可區分，故另外保留絕對剩餘 CPU。
        cpu = tuple(observation["cpu_remaining"])
        if len(cpu) != len(ids) or any(not isfinite(v) for v in cpu):
            raise ValueError("cpu_remaining must align with node_ids and be finite")
        order = {node: i / max_steps for i, node in enumerate(path)}
        rows = [tuple(row) + (available / cfg.cpu_demand_scale, order.get(n, 0.0)) + tuple(context)
                for n, row, available in zip(ids, nodes, cpu)]
        if any(not isfinite(v) for row in rows for v in row) or any(not isfinite(v) for row in attrs for v in row):
            raise ValueError("Input features must be finite")
        x = torch.tensor(rows, dtype=dtype, device=device)
        edge_attr = torch.tensor(attrs, dtype=dtype, device=device).reshape(-1, cfg.edge_dim)
        # 即使完全無邊也維持 [2,0] 與 [0,2]，不產生模糊的一維空張量。
        edge_index = torch.tensor(edges, dtype=torch.long, device=device).reshape(2, -1)
        if not bool(torch.isfinite(x).all()) or not bool(torch.isfinite(edge_attr).all()):
            raise ValueError("Input features overflowed the selected tensor dtype")
        return GATInputs(
            x=x, edge_index=edge_index, edge_attr=edge_attr,
            position=torch.tensor(position, dtype=dtype, device=device),
            action_mask=torch.tensor(observation["action_mask"], dtype=torch.bool, device=device),
            feasible_action_mask=torch.tensor(observation["feasible_action_mask"], dtype=torch.bool, device=device),
            request_features=torch.tensor(context, dtype=dtype, device=device),
            node_ids=ids, node_feature_names=self.node_feature_names,
            edge_feature_names=self.EDGE_NAMES, request_feature_names=self.request_feature_names,
        )
