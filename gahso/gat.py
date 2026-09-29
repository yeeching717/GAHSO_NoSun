"""含邊特徵的 Graph Attention Network，對應論文公式 (19)–(22)。

依公式 (21) 使用鄰居 softmax（原文敘述的 sigmoid 與公式不一致）。
邊特徵參與注意力打分；公式 (22) 的 message 本身是投影後的鄰居節點特徵。
"""
from dataclasses import dataclass
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from .features import GATInputs, GATFeatureBuilder
from .gat_config import GATConfig, GATFeatureConfig


@dataclass(frozen=True)
class AttentionWeights:
    """含模型自迴圈的訊息邊 [2,M+N] 與每頭注意力 [M+N,heads]。"""
    edge_index: Tensor
    alpha: Tensor


class EdgeAwareGATLayer(nn.Module):
    """單層 GAT；edge_index = [sender j, receiver i]，在 receiver 的鄰域正規化。

    每個頭獨立做節點／邊投影與注意力，多頭輸出取平均以維持 out_channels。
    預設單頭對應論文公式，多頭是可選擴充。
    """

    def __init__(self, in_channels: int, edge_channels: int, out_channels: int, *,
                 heads: int = 1, negative_slope: float = 0.2):
        super().__init__()
        if any(type(v) is not int or v < 1 for v in (in_channels, edge_channels, out_channels, heads)):
            raise ValueError("Channel counts and heads must be positive integers")
        if not 0 < negative_slope < 1:
            raise ValueError("negative_slope must be between 0 and 1")
        self.in_channels = in_channels
        self.edge_channels = edge_channels
        self.out_channels = out_channels
        self.heads = heads
        self.negative_slope = negative_slope
        # 式 (19)/(20)：分開學習節點與邊的投影，無 bias 以保持公式形式。
        self.node_projection = nn.Linear(in_channels, heads * out_channels, bias=False)
        self.edge_projection = nn.Linear(edge_channels, heads * out_channels, bias=False)
        self.attention = nn.Parameter(torch.empty(heads, 3 * out_channels))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.node_projection.weight)
        nn.init.xavier_uniform_(self.edge_projection.weight)
        nn.init.xavier_uniform_(self.attention)

    def _validate(self, x: Tensor, edge_index: Tensor, edge_attr: Tensor):
        if x.ndim != 2 or x.shape[0] == 0 or x.shape[1] != self.in_channels:
            raise ValueError("x must have shape [N, in_channels] with N > 0")
        if edge_index.ndim != 2 or edge_index.shape[0] != 2 or edge_index.dtype != torch.long:
            raise ValueError("edge_index must be torch.long with shape [2, M]")
        if edge_attr.ndim != 2 or edge_attr.shape != (edge_index.shape[1], self.edge_channels):
            raise ValueError("edge_attr must align with edge_index")
        if x.dtype not in (torch.float32, torch.float64) or edge_attr.dtype != x.dtype:
            raise ValueError("Node and edge features must use the same float32/float64 dtype")
        if x.device != edge_index.device or x.device != edge_attr.device:
            raise ValueError("All graph tensors must be on the same device")
        if self.node_projection.weight.device != x.device or self.node_projection.weight.dtype != x.dtype:
            raise ValueError("Move model and graph to the same device/dtype before calling forward")
        if edge_index.numel() and (bool((edge_index < 0).any()) or bool((edge_index >= x.shape[0]).any())):
            raise ValueError("Edge endpoint is outside the node matrix")
        if not bool(torch.isfinite(x).all()) or not bool(torch.isfinite(edge_attr).all()):
            raise ValueError("Graph features must be finite")
        nonself = edge_index[:, edge_index[0] != edge_index[1]]
        keys = nonself[0] * x.shape[0] + nonself[1]
        if torch.unique(keys).numel() != keys.numel():
            raise ValueError("Duplicate non-self edges would double-count attention")

    def forward(self, x: Tensor, edge_index: Tensor, edge_attr: Tensor, *,
                return_attention: bool = False):
        self._validate(x, edge_index, edge_attr)
        n = x.shape[0]
        # 式 (22) 的 alpha_ii：移除輸入自迴圈後，每個節點恰好加一次。
        # 零邊特徵是建模選擇；此邊只存在模型內，不會扣除 MDP 的頻寬。
        keep = edge_index[0] != edge_index[1]
        loops = torch.arange(n, device=x.device)
        augmented_index = torch.cat((edge_index[:, keep], torch.stack((loops, loops))), dim=1)
        augmented_attr = torch.cat((edge_attr[keep], edge_attr.new_zeros((n, self.edge_channels))), dim=0)
        sender, receiver = augmented_index
        h = self.node_projection(x).reshape(n, self.heads, self.out_channels)
        e = self.edge_projection(augmented_attr).reshape(-1, self.heads, self.out_channels)
        # 式 (21)：a^T [h_i || h_j || e_ij]，i 是接收端，j 是鄰居來源。
        joined = torch.cat((h[receiver], h[sender], e), dim=-1)
        logits = F.leaky_relu((joined * self.attention).sum(dim=-1), self.negative_slope)

        # 穩定的分組 softmax；只在同一接收端的「鄰居 + 自身」之間正規化。
        # 使用稀疏 edge_index，避免建立 N×N 的密集注意力矩陣。
        group = receiver[:, None].expand(-1, self.heads)
        maximum = logits.new_full((n, self.heads), -torch.inf)
        maximum.scatter_reduce_(0, group, logits.detach(), reduce="amax", include_self=True)
        numerator = torch.exp(logits - maximum[receiver])
        denominator = logits.new_zeros((n, self.heads))
        denominator.scatter_add_(0, group, numerator)
        alpha = numerator / denominator[receiver]

        # 式 (22)：注意力加權的是節點 message，邊資訊透過 alpha 影響聚合。
        messages = h[sender] * alpha.unsqueeze(-1)
        output = h.new_zeros((n, self.heads, self.out_channels))
        output.index_add_(0, receiver, messages)
        output = output.mean(dim=1)
        if return_attention:
            return output, AttentionWeights(augmented_index, alpha)
        return output


class GAHSOGAT(nn.Module):
    """GAT 特徵提取器：輸入一張 GATInputs，輸出 [N, hidden_dim]。

    尚不包含 Actor/Critic。兩層、層間 ELU 與 dropout 為可調實作選擇。
    每層都對原始邊特徵重新投影；不同層不共用投影參數。
    """
    def __init__(self, feature_config: GATFeatureConfig | None = None,
                 config: GATConfig | None = None):
        super().__init__()
        self.feature_config = feature_config or GATFeatureConfig()
        self.config = config or GATConfig()
        self.node_feature_names = GATFeatureBuilder(self.feature_config).node_feature_names
        self.layers = nn.ModuleList([
            EdgeAwareGATLayer(
                self.feature_config.node_dim if i == 0 else self.config.hidden_dim,
                self.feature_config.edge_dim, self.config.hidden_dim,
                heads=self.config.heads, negative_slope=self.config.negative_slope,
            )
            for i in range(self.config.num_layers)
        ])

    def forward(self, graph: GATInputs, *, return_attention: bool = False):
        if graph.node_feature_names != self.node_feature_names:
            raise ValueError("Feature builder and model use different node schemas")
        if graph.edge_feature_names != GATFeatureBuilder.EDGE_NAMES:
            raise ValueError("Unexpected edge feature schema")
        h = graph.x
        attention = []
        for i, layer in enumerate(self.layers):
            if return_attention:
                h, weights = layer(h, graph.edge_index, graph.edge_attr, return_attention=True)
                attention.append(weights)
            else:
                h = layer(h, graph.edge_index, graph.edge_attr)
            if i < len(self.layers) - 1:
                h = F.dropout(F.elu(h), p=self.config.dropout, training=self.training)
        return (h, tuple(attention)) if return_attention else h
