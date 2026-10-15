"""Algorithm 4 的請求層級更新：完整編排、VNF 放置、終止獎勵後才 step。"""
from dataclasses import dataclass
from time import perf_counter
import torch
from torch import Tensor
from .actor_critic import ActorCriticAgent
from .mdp import SFCOrchestrationEnv
from .rollout import RequestRollout, RolloutCollector
from .services import SFCRequest
from .training_config import TrainingConfig


@dataclass(frozen=True)
class RequestLosses:
    actor: Tensor
    critic: Tensor
    entropy: Tensor
    policy_steps: int


def normalize_advantages(advantages: Tensor, epsilon: float = 1e-6) -> Tensor:
    """逐請求標準化；單步/常數軌跡保留方向，避免扣平均後失去學習訊號。"""
    advantages = advantages.detach()
    std = advantages.std(unbiased=False)
    if advantages.numel() > 1 and std.item() > epsilon:
        return (advantages - advantages.mean()) / std.clamp_min(epsilon)
    return advantages / advantages.abs().max().clamp_min(1.0)


def request_losses(rollout: RequestRollout, entropy_coefficient: float = 0.0,
                   advantage_epsilon: float = 1e-6) -> RequestLosses:
    """平均各步 loss，讓不同長度請求的梯度尺度一致。

    advantage=target-V；最小化 -log_pi*adv 等價於提高正 advantage 動作機率。
    原文式 (23) 使用相反符號 V-target；不照搬其與 ascent 敘述矛盾的方向。
    """
    TrainingConfig(entropy_coefficient=entropy_coefficient)
    steps = rollout.transitions
    if not steps or not (steps[-1].terminated or steps[-1].truncated):
        raise ValueError("A complete request rollout is required")
    if any(t.terminated or t.truncated for t in steps[:-1]):
        raise ValueError("Rollout contains transitions after termination")
    if any(not t.value.requires_grad for t in steps):
        raise ValueError("Collect with track_grad=True for training")
    policy = [t for t in steps if t.action is not None]
    if any(t.log_prob is None or t.entropy is None or not t.log_prob.requires_grad for t in policy):
        raise ValueError("Policy transitions require differentiable log probabilities")
    critic = torch.stack([.5 * (t.value - t.target.detach()).square() for t in steps]).mean()
    zero = critic.detach().new_zeros(())
    entropy = torch.stack([t.entropy for t in policy]).mean() if policy else zero
    # advantage 與 bootstrap target 都固定在收集時，不讓 Actor loss 回傳至 Critic。
    advantages = normalize_advantages(torch.stack([t.advantage for t in policy]), advantage_epsilon) if policy else zero
    actor = (-(torch.stack([t.log_prob for t in policy]) * advantages).mean()
             - entropy_coefficient * entropy) if policy else zero
    return RequestLosses(actor, critic, entropy, len(policy))


def batch_request_losses(rollouts, entropy_coefficient=0.0, advantage_epsilon=1e-6):
    """Normalize policy advantages across complete requests in one update batch."""
    transitions = [t for rollout in rollouts for t in rollout.transitions]
    if not transitions or any(not (r.transitions[-1].terminated or r.transitions[-1].truncated)
                              for r in rollouts):
        raise ValueError("Complete request rollouts are required")
    policy = [t for t in transitions if t.action is not None]
    critic = torch.stack([.5 * (t.value - t.target.detach()).square()
                          for t in transitions]).mean()
    zero = critic.detach().new_zeros(())
    entropy = torch.stack([t.entropy for t in policy]).mean() if policy else zero
    advantages = normalize_advantages(torch.stack([t.advantage for t in policy]),
                                      advantage_epsilon) if policy else zero
    actor = (-(torch.stack([t.log_prob for t in policy]) * advantages).mean()
             - entropy_coefficient * entropy) if policy else zero
    return RequestLosses(actor, critic, entropy, len(policy))


class A2CTrainer:
    """一次 train_request 僅消費一次新請求；不提供重複更新舊軌跡的介面。

    完成 collect 前沒有 backward/optimizer.step；成功、延遲違反、K 用盡均更新。
    無任何動作的孤立起點只更新 Critic，不製造 Actor 梯度。
    """
    def __init__(self, env: SFCOrchestrationEnv, agent: ActorCriticAgent,
                 config: TrainingConfig | None = None, *, seed: int = 11):
        self.env, self.agent = env, agent
        self.config = config or TrainingConfig()
        self.collector = RolloutCollector(env, agent, seed=seed, reward_scale=self.config.reward_scale)
        self.actor_optimizer = torch.optim.Adam(agent.actor.parameters(), lr=self.config.actor_lr)
        self.critic_optimizer = torch.optim.Adam(agent.critic.parameters(), lr=self.config.critic_lr)
        self.requests_updated = 0
        self.actor_updates = 0

    def train_request(self, request: SFCRequest, *, include_trajectory: bool = False) -> dict:
        """採目前策略隨機取樣，保留背景資源；回傳純 Python 報告，不保留 autograd 圖。"""
        self.actor_optimizer.zero_grad(set_to_none=True)
        self.critic_optimizer.zero_grad(set_to_none=True)
        started = perf_counter()
        try:
            rollout = self.collector.collect(request, track_grad=True)
            losses = request_losses(rollout, self.config.entropy_coefficient, self.config.advantage_epsilon)
            if not all(bool(torch.isfinite(x)) for x in (losses.actor, losses.critic, losses.entropy)):
                raise FloatingPointError("Nonfinite request loss; no optimizer update performed")
            (losses.actor + losses.critic).backward()
            # 先檢查兩個網路，避免 Actor 已更新才發現 Critic 梯度無效。
            for model in (self.agent.actor, self.agent.critic):
                if any(p.grad is not None and not bool(torch.isfinite(p.grad).all())
                       for p in model.parameters()):
                    raise FloatingPointError("Nonfinite gradients; no optimizer update performed")
            if self.config.max_grad_norm is not None:
                for model in (self.agent.actor, self.agent.critic):
                    torch.nn.utils.clip_grad_norm_(model.parameters(), self.config.max_grad_norm,
                                                   error_if_nonfinite=True)
            final = rollout.transitions[-1]
            td = torch.stack([t.advantage.detach() for t in rollout.transitions])
            policy_adv = torch.stack([t.advantage.detach() for t in rollout.transitions
                                      if t.action is not None]) if losses.policy_steps else td.new_zeros(1)
            feasible = sum(int(t.graph.feasible_action_mask.sum()) for t in rollout.transitions)
            neighbors = sum(int(t.graph.action_mask.sum()) for t in rollout.transitions)
            report = {
                "request_id": request.id, "accepted": rollout.accepted,
                "steps": len(rollout.transitions), "policy_steps": losses.policy_steps,
                "total_reward": rollout.total_reward, "terminal_reward": final.reward,
                "terminal_info": final.info, "path": final.next_state["path"],
                "failure_category": final.info["failure_category"],
                "initial_route_feasible": final.info["initial_diagnostics"]["route_feasible"],
                "actor_loss": float(losses.actor.detach()), "critic_loss": float(losses.critic.detach()),
                "entropy": float(losses.entropy.detach()),
                "mean_advantage": float(policy_adv.mean()),
                "std_advantage": float(policy_adv.std(unbiased=False)),
                "mean_td_error": float(td.mean()), "mean_abs_td_error": float(td.abs().mean()),
                "reward_before_normalization": rollout.total_reward,
                "reward_after_normalization": rollout.total_reward / self.config.reward_scale,
                "feasible_action_count": feasible, "neighbor_action_count": neighbors,
                "feasible_action_ratio": feasible / neighbors if neighbors else 0.0,
                "invalid_action_count": sum(t.action is not None and
                    not bool(t.graph.feasible_action_mask[t.action]) for t in rollout.transitions),
            }
            if include_trajectory:
                report["trajectory"] = rollout.to_dict()["transitions"]
            # 每個請求只有這一處更新；請求內的所有跳使用相同參數。
            if losses.policy_steps:
                self.actor_optimizer.step()
                self.actor_updates += 1
            self.critic_optimizer.step()
            synced = self.agent.notify_critic_updated()
            self.requests_updated += 1
            report.update(requests_updated=self.requests_updated, actor_updates=self.actor_updates,
                          critic_updates=int(self.agent.critic_updates.item()), target_synced=synced,
                          actor_updated=bool(losses.policy_steps))
            report["training_seconds"] = perf_counter() - started
            return report
        finally:
            # 避免下個請求累加上一個請求梯度，並釋放不再需要的梯度。
            self.actor_optimizer.zero_grad(set_to_none=True)
            self.critic_optimizer.zero_grad(set_to_none=True)

    def train_rollouts(self, rollouts, *, elapsed_seconds=None):
        """Apply one optimizer update to a batch of already completed requests."""
        rollouts = tuple(rollouts)
        if not rollouts:
            return []
        self.actor_optimizer.zero_grad(set_to_none=True)
        self.critic_optimizer.zero_grad(set_to_none=True)
        try:
            losses = batch_request_losses(rollouts, self.config.entropy_coefficient,
                                          self.config.advantage_epsilon)
            if not all(bool(torch.isfinite(x)) for x in (losses.actor, losses.critic, losses.entropy)):
                raise FloatingPointError("Nonfinite batch loss")
            (losses.actor + losses.critic).backward()
            for model in (self.agent.actor, self.agent.critic):
                if any(p.grad is not None and not bool(torch.isfinite(p.grad).all()) for p in model.parameters()):
                    raise FloatingPointError("Nonfinite gradients")
                if self.config.max_grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), self.config.max_grad_norm,
                                                   error_if_nonfinite=True)
            if losses.policy_steps:
                self.actor_optimizer.step()
                self.actor_updates += 1
            self.critic_optimizer.step()
            synced = self.agent.notify_critic_updated()
            self.requests_updated += len(rollouts)
            seconds = elapsed_seconds if elapsed_seconds is not None else 0.0
            reports = []
            for rollout in rollouts:
                final = rollout.transitions[-1]
                td = torch.stack([t.advantage.detach() for t in rollout.transitions])
                policy = [t for t in rollout.transitions if t.action is not None]
                policy_adv = torch.stack([t.advantage.detach() for t in policy]) if policy else td.new_zeros(1)
                feasible = sum(int(t.graph.feasible_action_mask.sum()) for t in rollout.transitions)
                neighbors = sum(int(t.graph.action_mask.sum()) for t in rollout.transitions)
                initial = final.info["initial_diagnostics"]
                reports.append({
                    "request_id": rollout.request_id, "accepted": rollout.accepted,
                    "steps": len(rollout.transitions), "policy_steps": len(policy),
                    "total_reward": rollout.total_reward, "terminal_reward": final.reward,
                    "terminal_info": final.info, "path": final.next_state["path"],
                    "failure_category": final.info["failure_category"],
                    "initial_route_feasible": initial["route_feasible"],
                    "initially_feasible": initial["route_feasible"] and initial["endpoint_cpu_feasible"],
                    "reached_destination": final.next_state["path"][-1] == final.state["request"]["destination"],
                    "no_action_termination": final.action is None,
                    "actor_loss": float(losses.actor.detach()), "critic_loss": float(losses.critic.detach()),
                    "entropy": float(losses.entropy.detach()),
                    "mean_advantage": float(policy_adv.mean()),
                    "std_advantage": float(policy_adv.std(unbiased=False)),
                    "mean_td_error": float(td.mean()), "mean_abs_td_error": float(td.abs().mean()),
                    "reward_before_normalization": rollout.total_reward,
                    "reward_after_normalization": rollout.total_reward / self.config.reward_scale,
                    "feasible_action_count": feasible, "neighbor_action_count": neighbors,
                    "feasible_action_ratio": feasible / neighbors if neighbors else 0.0,
                    "invalid_action_count": sum(t.action is not None and not bool(
                        t.graph.feasible_action_mask[t.action]) for t in rollout.transitions),
                    "requests_updated": self.requests_updated, "actor_updates": self.actor_updates,
                    "critic_updates": int(self.agent.critic_updates.item()), "target_synced": synced,
                    "actor_updated": bool(losses.policy_steps),
                    "training_seconds": seconds / len(rollouts),
                })
            return reports
        finally:
            self.actor_optimizer.zero_grad(set_to_none=True)
            self.critic_optimizer.zero_grad(set_to_none=True)
