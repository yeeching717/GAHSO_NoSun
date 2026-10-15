"""GAT 設定獨立於 PyTorch，讓原有基準模擬仍可只用標準函式庫執行。"""
from dataclasses import dataclass
from math import isfinite


@dataclass(frozen=True)
class GATFeatureConfig:
    """原始特徵編碼的補充假設；超長 VNF 串列明確拒絕，絕不默默截斷。"""
    max_vnfs: int = 5
    cpu_demand_scale: float = 200.0
    bandwidth_demand_scale_mhz: float = 200.0
    delay_scale_s: float = 0.1

    def __post_init__(self):
        if type(self.max_vnfs) is not int or self.max_vnfs != 5:
            raise ValueError("max_vnfs must be 5 for the single-service feature schema")
        scales = (self.cpu_demand_scale, self.bandwidth_demand_scale_mhz, self.delay_scale_s)
        if not all(isfinite(x) and x > 0 for x in scales):
            raise ValueError("Feature scales must be finite and positive")

    @property
    def request_dim(self) -> int:
        # 頻寬、延遲上限、VNF 數量、剩餘延遲、剩餘步數，以及有序 CPU 與 padding mask。
        return 5 + 2 * self.max_vnfs

    @property
    def node_dim(self) -> int:
        # MDP 8 維（含延遲下界/可達性）+ CPU 1 維 + 路徑順序 1 維 + 服務需求。
        return 10 + self.request_dim

    @property
    def edge_dim(self) -> int:
        return 2


@dataclass(frozen=True)
class GATConfig:
    """128 維依論文；2 層、LeakyReLU 斜率及可選多頭為明確可調設定。"""
    hidden_dim: int = 128
    num_layers: int = 2
    heads: int = 1
    negative_slope: float = 0.2
    dropout: float = 0.0

    def __post_init__(self):
        if any(type(v) is not int or v < 1 for v in (self.hidden_dim, self.num_layers, self.heads)):
            raise ValueError("hidden_dim, num_layers and heads must be positive integers")
        if not isfinite(self.negative_slope) or not 0 < self.negative_slope < 1:
            raise ValueError("negative_slope must be between 0 and 1")
        if not isfinite(self.dropout) or not 0 <= self.dropout < 1:
            raise ValueError("dropout must be in [0, 1)")
