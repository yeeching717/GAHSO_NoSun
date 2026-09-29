"""Actor/Critic 超參數，獨立於 PyTorch 以維持基準模擬的輕量相依。"""
from dataclasses import dataclass
from math import isfinite


@dataclass(frozen=True)
class AgentConfig:
    """三層 MLP 的兩個隱藏寬度為補充設定；discount 依 Table I。

    target_sync_interval 計算「Critic 真正更新」次數，絕不把推論步數當更新。
    本階段未內建 optimizer；更新器未來在每次更新後呼叫 notify_critic_updated。
    """
    hidden_dim1: int = 128
    hidden_dim2: int = 64
    discount: float = 0.9999
    target_sync_interval: int = 10

    def __post_init__(self):
        if any(type(v) is not int or v <= 0
               for v in (self.hidden_dim1, self.hidden_dim2, self.target_sync_interval)):
            raise ValueError("MLP widths and target_sync_interval must be positive integers")
        if not isfinite(self.discount) or not 0 <= self.discount <= 1:
            raise ValueError("discount must be in [0, 1]")
