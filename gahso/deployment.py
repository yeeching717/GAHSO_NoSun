"""部署只載入 Actor，凍結推論；候選計畫不修改正式資源帳本。"""
from copy import deepcopy
from dataclasses import asdict, replace
from pathlib import Path
import torch
from .actor_critic import Actor
from .agent_config import AgentConfig
from .gat_config import GATConfig, GATFeatureConfig
from .features import GATFeatureBuilder
from .mdp import MDPConfig, SFCOrchestrationEnv
from .resources import Allocation


class FrozenTopology:
    """一次編排固定在當前物理快照，不能自行恢復故障鏈路。"""
    def __init__(self, snapshot):
        self.value = snapshot

    def snapshot(self, time_s):
        return replace(self.value, time_s=time_s)


class DeploymentPolicy:
    def __init__(self, actor: Actor, mdp_config: MDPConfig | None = None, *,
                 deterministic: bool = True, seed: int = 11):
        self.actor = deepcopy(actor).eval().requires_grad_(False)
        self.mdp_config = mdp_config or MDPConfig()
        self.deterministic = deterministic
        self.builder = GATFeatureBuilder(actor.feature_config)
        self.generator = torch.Generator(device=next(self.actor.parameters()).device).manual_seed(seed)
        self.last_report = None

    @torch.inference_mode()
    def __call__(self, manager, request):
        if len(manager.snapshot.satellites) != self.actor.num_nodes:
            raise ValueError("Checkpoint node count does not match deployment topology")
        if request.id in manager.records:
            raise ValueError("Duplicate request ID")
        if not request.arrival_s <= manager.time_s < request.expires_s:
            raise ValueError("Request is outside its lifetime")
        env = SFCOrchestrationEnv(FrozenTopology(manager.snapshot), self.mdp_config)
        env.resources = manager.planning_copy()
        state, _ = env.start_request(request)
        reference = next(self.actor.parameters())
        total_reward = 0.0
        while True:
            graph = self.builder.build(state, device=reference.device, dtype=reference.dtype)
            if bool(graph.feasible_action_mask.any()):
                distribution = self.actor.distribution(graph)
                action = (distribution.probs.argmax() if self.deterministic else
                          torch.multinomial(distribution.probs, 1, generator=self.generator)[0])
                state, reward, done, truncated, info = env.step(int(action))
            else:
                state, reward, done, truncated, info = env.finish_without_action()
            total_reward += reward
            if done or truncated:
                break
        initial = info["initial_diagnostics"]
        self.last_report = {"reward": total_reward, "steps": env.steps, "policy_steps": env.steps,
            "info": info, "failure_category": info["failure_category"],
            "initially_feasible": initial["route_feasible"] and initial["endpoint_cpu_feasible"],
            "no_action_termination": bool(info.get("no_action_reasons")),
            "endpoint_diagnostics": {k: initial[k] for k in (
                "endpoint_placement_mode", "source_endpoint_cpu_demand",
                "destination_endpoint_cpu_demand", "source_endpoint_available_cpu",
                "destination_endpoint_available_cpu")}}
        return Allocation(**info["allocation"]) if info["accepted"] else None


def save_policy(path, agent, mdp_config: MDPConfig | None = None):
    """純 state_dict＋架構／特徵／MDP 設定；不序列化任意 Python 模型物件。"""
    actor = agent.actor if hasattr(agent, "actor") else agent
    payload = {"version": 2, "feature_schema": "single_sfc_vnf5_delay_bound", "num_nodes": actor.num_nodes,
               "feature_config": asdict(actor.feature_config), "gat_config": asdict(actor.gat_config),
               "agent_config": asdict(actor.config), "mdp_config": asdict(mdp_config or MDPConfig()),
               "dtype": str(next(actor.parameters()).dtype).split(".")[-1],
               "state_dict": {k: v.detach().cpu() for k, v in actor.state_dict().items()}}
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def load_policy(path, *, device="cpu", deterministic=True, seed=11, expected_num_nodes=None):
    data = torch.load(path, map_location="cpu", weights_only=True)
    if expected_num_nodes is not None and data.get("num_nodes") != expected_num_nodes:
        raise ValueError(f"Checkpoint has {data.get('num_nodes')} satellites; expected {expected_num_nodes}. Retrain the model.")
    if (data.get("version") != 2 or data.get("feature_schema") != "single_sfc_vnf5_delay_bound"
            or data["dtype"] not in ("float32", "float64")):
        raise ValueError("Incompatible checkpoint feature schema; retrain for single-service 25-satellite configuration")
    # 載入不可擾動訓練程式的全域 RNG。
    with torch.random.fork_rng(devices=[]):
        actor = Actor(data["num_nodes"], GATFeatureConfig(**data["feature_config"]),
                      GATConfig(**data["gat_config"]), AgentConfig(**data["agent_config"]))
    actor = actor.to(device=device, dtype=getattr(torch, data["dtype"]))
    actor.load_state_dict(data["state_dict"], strict=True)
    return DeploymentPolicy(actor, MDPConfig.from_dict(data["mdp_config"]),
                            deterministic=deterministic, seed=seed)
