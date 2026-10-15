"""逐跳互動：將 Actor、Critic、Target Critic 與 MDP 的真實獎勵接在一起。"""
from contextlib import contextmanager
from dataclasses import dataclass
from math import isfinite
import torch
from torch import Tensor
from .actor_critic import ActorCriticAgent
from .features import GATFeatureBuilder, GATInputs
from .mdp import SFCOrchestrationEnv
from .services import SFCRequest


def td_target(reward: float, next_value: Tensor, terminated: bool, discount: float) -> Tensor:
    """r + mu*(1-done)*V_target(s_next)；target 不參與梯度。

    論文的 K 用盡屬 terminated，不能 bootstrap。
    真正的外部 truncated（如未來額外時間限制）則仍可使用下一狀態價值。
    """
    if not isfinite(reward) or not isfinite(discount) or not 0 <= discount <= 1:
        raise ValueError("Invalid reward or discount")
    if next_value.ndim != 0:
        raise ValueError("next_value must be a scalar")
    if terminated:
        # 不以 0*nan/inf 假裝遮罩；終止時完全忽略 next_value。
        return next_value.new_tensor(reward)
    if not bool(torch.isfinite(next_value)):
        raise ValueError("Nonterminal next value must be finite")
    return next_value.new_tensor(reward) + discount * next_value.detach()


@dataclass(frozen=True)
class Transition:
    """保存真實轉移；action=None 表示無可行動作，reward 原值與 target 縮放單位分開。"""
    state: dict
    next_state: dict
    graph: GATInputs
    next_graph: GATInputs
    action: int | None
    action_node_id: int | None
    reward: float
    terminated: bool
    truncated: bool
    log_prob: Tensor | None
    entropy: Tensor | None
    probabilities: Tensor | None
    value: Tensor
    next_value: Tensor
    target: Tensor
    advantage: Tensor
    info: dict

    def to_dict(self, *, include_states: bool = True) -> dict:
        def number(tensor):
            return None if tensor is None else float(tensor.detach())
        result = {
            "action": self.action, "action_node_id": self.action_node_id, "reward": self.reward,
            "terminated": self.terminated, "truncated": self.truncated,
            "log_prob": number(self.log_prob), "entropy": number(self.entropy),
            "probabilities": None if self.probabilities is None else self.probabilities.detach().cpu().tolist(),
            "value": number(self.value), "next_value": number(self.next_value),
            "td_target": number(self.target), "advantage": number(self.advantage),
            "paper_td_error": number(self.value.detach() - self.target), "info": self.info,
        }
        if include_states:
            result.update(state=self.state, next_state=self.next_state)
        return result


@dataclass(frozen=True)
class RequestRollout:
    request_id: str
    transitions: tuple[Transition, ...]

    @property
    def total_reward(self) -> float:
        return sum(step.reward for step in self.transitions)

    @property
    def accepted(self) -> bool:
        return bool(self.transitions[-1].info["accepted"])

    def to_dict(self, *, include_states: bool = True) -> dict:
        return {"request_id": self.request_id, "total_reward": self.total_reward,
                "accepted": self.accepted, "policy_steps": sum(t.action is not None for t in self.transitions),
                "transitions": [t.to_dict(include_states=include_states) for t in self.transitions]}


@contextmanager
def _collection_mode(agent: ActorCriticAgent, track_grad: bool):
    """rollout 使用確定性網路前向（關閉 dropout），動作仍可隨機取樣。

    eval 不等於 no_grad；track_grad=True 時保留 Actor/Critic 圖供請求結束後更新。
    """
    actor_training, critic_training = agent.actor.training, agent.critic.training
    agent.actor.eval()
    agent.critic.eval()
    try:
        with torch.set_grad_enabled(track_grad):
            yield
    finally:
        agent.actor.train(actor_training)
        agent.critic.train(critic_training)


class RolloutCollector:
    """每個請求內不改動網路參數，完成後回傳完整軌跡；不自動 optimizer.step。"""
    def __init__(self, env: SFCOrchestrationEnv, agent: ActorCriticAgent, *, seed: int = 11,
                 reward_scale: float = 1.0):
        if not isfinite(reward_scale) or reward_scale <= 0:
            raise ValueError("reward_scale must be finite and positive")
        self.reward_scale = reward_scale
        self.env = env
        self.agent = agent
        self.builder = GATFeatureBuilder(agent.actor.feature_config)
        if len(env.node_ids) != agent.actor.num_nodes:
            raise ValueError("Environment and agent have different node counts")
        reference = next(agent.actor.parameters())
        self.generator = torch.Generator(device=reference.device).manual_seed(seed)

    def collect(self, request: SFCRequest, *, reset_network: bool = False,
                deterministic: bool = False, track_grad: bool = False) -> RequestRollout:
        """預設 start_request 保留背景資源；reset_network=True 才清空環境。

        對無任何可行動作的狀態，明確記錄 action=None 的失敗終止，
        不偽造動作，也不對全 -inf 的 logits 計算 softmax。
        """
        agent, env = self.agent, self.env
        reference = next(agent.actor.parameters())
        if self.generator.device != reference.device:
            raise ValueError("Recreate collector after moving the agent to a different device")
        # 在修改環境之前檢查 VNF 長度，避免開始了卻無法編碼的請求。
        if len(request.vnfs) > self.builder.config.max_vnfs:
            raise ValueError("Request exceeds max_vnfs; increase feature configuration")
        state, _ = env.reset(request=request) if reset_network else env.start_request(request)
        trajectory = []
        with _collection_mode(agent, track_grad):
            while True:
                graph = self.builder.build(state, device=reference.device, dtype=reference.dtype)
                value = agent.critic(graph)
                if bool(graph.feasible_action_mask.any()):
                    sample = agent.act(graph, deterministic=deterministic, generator=self.generator)
                    next_state, reward, terminated, truncated, info = env.step(sample.action)
                    action, node = sample.action, graph.node_ids[sample.action]
                else:
                    sample = None
                    next_state, reward, terminated, truncated, info = env.finish_without_action()
                    action = node = None
                next_graph = self.builder.build(next_state, device=reference.device, dtype=reference.dtype)
                # 成功或失敗終止都不 bootstrap；不用在全遮罩狀態再次呼叫 Actor。
                next_value = value.detach().new_zeros(()) if terminated else agent.target_critic(next_graph)
                target = td_target(reward / self.reward_scale, next_value, terminated, agent.config.discount)
                # 原文式 (23) 是 V-target；常見 policy advantage 則採 target-V。
                advantage = target - value.detach()
                trajectory.append(Transition(
                    state, next_state, graph, next_graph, action, node, reward, terminated, truncated,
                    sample.log_prob if sample else None, sample.entropy if sample else None,
                    sample.probabilities if sample else None, value, next_value, target, advantage, info,
                ))
                if terminated or truncated:
                    break
                state = next_state
        return RequestRollout(request.id, tuple(trajectory))
