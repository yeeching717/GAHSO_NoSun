"""每個請求更新一次的訓練設定；不引入 PyTorch 相依。"""
from dataclasses import dataclass
from math import isfinite


@dataclass(frozen=True)
class TrainingConfig:
    # 下列 learning rate 為可調工程預設，非聲稱論文 Table I 的精確重現值。
    actor_lr: float = 1e-4
    critic_lr: float = 1e-4
    entropy_coefficient: float = 0.01
    max_grad_norm: float | None = 5.0
    reward_scale: float = 5.0
    advantage_epsilon: float = 1e-6
    update_requests_per_batch: int = 8

    def __post_init__(self):
        for value in (self.actor_lr, self.critic_lr, self.reward_scale, self.advantage_epsilon):
            if not isfinite(value) or value <= 0:
                raise ValueError("Learning rates must be finite and positive")
        if not isfinite(self.entropy_coefficient) or self.entropy_coefficient < 0:
            raise ValueError("Entropy coefficient must be finite and nonnegative")
        if self.max_grad_norm is not None and (not isfinite(self.max_grad_norm) or self.max_grad_norm <= 0):
            raise ValueError("max_grad_norm must be finite and positive")
        if type(self.update_requests_per_batch) is not int or self.update_requests_per_batch <= 0:
            raise ValueError("update_requests_per_batch must be a positive integer")
