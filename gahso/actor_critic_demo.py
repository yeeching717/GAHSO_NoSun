"""真實 Actor 分布逐跳取樣示範；尚未訓練，因此不保證服務被接受。"""
from dataclasses import asdict
import torch
from .actor_critic import ActorCriticAgent
from .mdp import SFCOrchestrationEnv
from .network import LEOConstellation
from .rollout import RolloutCollector
from .services import SFCRequest, VNF


def run_actor_critic_demo(config) -> dict:
    env = SFCOrchestrationEnv(LEOConstellation(config.constellation), config.mdp,
                             cpu_load_weight=config.cpu_load_weight,
                             bandwidth_load_weight=config.bandwidth_load_weight)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(7)
        agent = ActorCriticAgent(config.constellation.satellites, config.gat_features,
                                config.gat, config.agent)
    collector = RolloutCollector(env, agent, seed=11)
    request = SFCRequest("actor-critic-demo", 0, 2,
                         (VNF("ingress", 50), VNF("process", 150), VNF("egress", 75)),
                         30, 0.1, 0, 30, "compute")
    rollout = collector.collect(request)
    before = env.resources.metrics()
    env.advance_time(request.expires_s)
    return {
        "mode": "actor_critic_demo", "trained": False,
        "agent_config": asdict(config.agent), "num_nodes": agent.actor.num_nodes,
        "actor_output_dim": agent.actor.num_nodes, "critic_output_dim": 1,
        "target_frozen": all(not p.requires_grad for p in agent.target_critic.parameters()),
        "target_equal_to_critic": all(torch.equal(a, b) for a, b in
            zip(agent.critic.parameters(), agent.target_critic.value_network.parameters())),
        "critic_updates": int(agent.critic_updates.item()),
        "request_id": request.id, "steps": len(rollout.transitions),
        "accepted": rollout.accepted, "total_reward": rollout.total_reward,
        "outcome": rollout.transitions[-1].info["outcome"],
        "path": rollout.transitions[-1].next_state["path"],
        "trajectory": rollout.to_dict()["transitions"],
        "before_expiry": before, "after_expiry": env.resources.metrics(),
    }
