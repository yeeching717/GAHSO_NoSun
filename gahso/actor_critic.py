"""GAHSO 的 Actor、Critic 與獨立 Target Critic；本模組不執行 optimizer 更新。"""
from copy import deepcopy
from dataclasses import dataclass
import torch
from torch import Tensor, nn
from torch.distributions import Categorical
from .agent_config import AgentConfig
from .features import GATInputs
from .gat import GAHSOGAT
from .gat_config import GATConfig, GATFeatureConfig


class NoAvailableActionsError(ValueError):
    """全遮罩時沒有合法 categorical distribution，必須由環境結束該請求。"""


class _GraphValueBase(nn.Module):
    """依第 VI-C 節：展平 GAT 輸出，串接目前位置 one-hot，再送入三層 MLP。

    Actor/Critic 各自擁有 GAT，讓兩組梯度與 Target 的同步範圍明確獨立。
    展平架構的 N 固定；變更衛星數時須重建 Actor/Critic。
    """
    def __init__(self, num_nodes: int, output_dim: int,
                 feature_config: GATFeatureConfig | None,
                 gat_config: GATConfig | None, config: AgentConfig | None):
        super().__init__()
        if type(num_nodes) is not int or num_nodes <= 0:
            raise ValueError("num_nodes must be a positive integer")
        self.num_nodes = num_nodes
        self.feature_config = feature_config or GATFeatureConfig()
        self.gat_config = gat_config or GATConfig()
        self.config = config or AgentConfig()
        self.encoder = GAHSOGAT(self.feature_config, self.gat_config)
        input_dim = num_nodes * (self.gat_config.hidden_dim + 1)
        self.head = nn.Sequential(
            nn.Linear(input_dim, self.config.hidden_dim1), nn.ReLU(),
            nn.Linear(self.config.hidden_dim1, self.config.hidden_dim2), nn.ReLU(),
            nn.Linear(self.config.hidden_dim2, output_dim),
        )

    def _representation(self, graph: GATInputs) -> Tensor:
        if len(graph.node_ids) != self.num_nodes or graph.x.shape[0] != self.num_nodes:
            raise ValueError("Graph size differs from the fixed Actor/Critic node count")
        if graph.position.shape != (self.num_nodes,):
            raise ValueError("Position vector must align with nodes")
        if graph.position.device != graph.x.device or graph.position.dtype != graph.x.dtype:
            raise ValueError("Position must match graph device and dtype")
        if not bool(((graph.position == 0) | (graph.position == 1)).all()) or graph.position.sum().item() not in (0, 1):
            raise ValueError("Position must be one-hot or all-zero for an idle graph")
        return torch.cat((self.encoder(graph).reshape(-1), graph.position))


class Actor(_GraphValueBase):
    """輸出每個 node_ids 索引的 logit；非鄰居設為 -inf 後才做 softmax。"""
    def __init__(self, num_nodes: int = 20, feature_config: GATFeatureConfig | None = None,
                 gat_config: GATConfig | None = None, config: AgentConfig | None = None):
        super().__init__(num_nodes, num_nodes, feature_config, gat_config, config)

    def raw_logits(self, graph: GATInputs) -> Tensor:
        logits = self.head(self._representation(graph))
        if not bool(torch.isfinite(logits).all()):
            raise ValueError("Actor produced nonfinite logits")
        return logits

    def forward(self, graph: GATInputs) -> Tensor:
        mask = graph.action_mask
        if mask.dtype != torch.bool or mask.shape != (self.num_nodes,) or mask.device != graph.x.device:
            raise ValueError("action_mask must be a boolean vector on the graph device")
        if not bool(mask.any()):
            raise NoAvailableActionsError("No neighbor action is available")
        return self.raw_logits(graph).masked_fill(~mask, -torch.inf)

    def distribution(self, graph: GATInputs) -> Categorical:
        """直接使用 logits 建立穩定分布；不使用更強的 feasible_action_mask。"""
        return Categorical(logits=self(graph))


class Critic(_GraphValueBase):
    """回傳純量 V(s)，包含 terminal state 也可估值，不依賴動作遮罩。"""
    def __init__(self, num_nodes: int = 20, feature_config: GATFeatureConfig | None = None,
                 gat_config: GATConfig | None = None, config: AgentConfig | None = None):
        super().__init__(num_nodes, 1, feature_config, gat_config, config)

    def forward(self, graph: GATInputs) -> Tensor:
        value = self.head(self._representation(graph)).squeeze(-1)
        if not bool(torch.isfinite(value)):
            raise ValueError("Critic produced a nonfinite value")
        return value


class TargetCritic(nn.Module):
    """Critic（含 GAT）完整深拷貝；無共享參數、無梯度、永遠維持 eval 模式。"""
    def __init__(self, critic: Critic):
        super().__init__()
        self.value_network = deepcopy(critic)
        self.requires_grad_(False)
        self.train(False)

    def train(self, mode: bool = True):
        # 即使外層 agent.train()，目標網路也不能開啟 dropout。
        super().train(False)
        return self

    @torch.no_grad()
    def forward(self, graph: GATInputs) -> Tensor:
        return self.value_network(graph)

    @torch.no_grad()
    def sync_from(self, critic: Critic):
        """Hard update：只在明確同步點把線上 Critic 的完整 state_dict 複製過來。"""
        target = self.value_network
        if (target.num_nodes != critic.num_nodes or target.feature_config != critic.feature_config
                or target.gat_config != critic.gat_config or target.config != critic.config):
            raise ValueError("Cannot synchronize critics with different configurations")
        target.load_state_dict(critic.state_dict())
        self.requires_grad_(False)
        self.train(False)


@dataclass(frozen=True)
class ActionSample:
    action: int
    log_prob: Tensor
    entropy: Tensor
    probabilities: Tensor


class ActorCriticAgent(nn.Module):
    """封裝三個獨立網路、動作取樣及 Target 同步計數。"""
    def __init__(self, num_nodes: int = 20, feature_config: GATFeatureConfig | None = None,
                 gat_config: GATConfig | None = None, config: AgentConfig | None = None):
        super().__init__()
        self.config = config or AgentConfig()
        self.actor = Actor(num_nodes, feature_config, gat_config, self.config)
        self.critic = Critic(num_nodes, feature_config, gat_config, self.config)
        self.target_critic = TargetCritic(self.critic)
        # 使用 buffer，使同步計數會隨 state_dict 保存／還原。
        self.register_buffer("critic_updates", torch.zeros((), dtype=torch.long))

    def act(self, graph: GATInputs, *, deterministic: bool = False,
            generator: torch.Generator | None = None) -> ActionSample:
        distribution = self.actor.distribution(graph)
        if deterministic:
            action = distribution.probs.argmax()
        else:
            # 專用 RNG 可重現 rollout，且不污染全域 torch RNG。
            action = torch.multinomial(distribution.probs, 1, generator=generator).squeeze(0)
        return ActionSample(int(action.item()), distribution.log_prob(action),
                            distribution.entropy(), distribution.probs)

    def sync_target(self):
        """立即同步，不計為一次 Critic 訓練更新。"""
        self.target_critic.sync_from(self.critic)

    @torch.no_grad()
    def notify_critic_updated(self) -> bool:
        """外部 optimizer.step() 完成後呼叫；達指定間隔時同步並回傳 True。"""
        self.critic_updates.add_(1)
        if self.critic_updates.item() % self.config.target_sync_interval == 0:
            self.sync_target()
            return True
        return False
