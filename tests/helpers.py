"""可執行的 MDP 介面示範；以最短跳數選動作，不宣稱為訓練後的 GAHSO。"""
from collections import deque
from gahso.mdp import SFCOrchestrationEnv
from gahso.network import LEOConstellation
from gahso.services import SFCRequest, VNF


def check_mdp_lifecycle(config) -> dict:
    """一個小型請求完整走過 reset→step→接受/拒絕→到期釋放。"""
    env = SFCOrchestrationEnv(LEOConstellation(config.constellation), config.mdp,
                             cpu_load_weight=config.cpu_load_weight,
                             bandwidth_load_weight=config.bandwidth_load_weight)
    request = SFCRequest("mdp-fixture", 0, 2, tuple(VNF(f"vnf-{i}", 75) for i in range(3)),
                         30, 0.1, 0, 30)
    observation, _ = env.reset(request=request)
    trajectory = []
    done = False
    while not done:
        # 僅供示範：BFS 距離作為啟發式分數；正式策略由 Actor 負責；此處只驗證 MDP 介面。
        distances = {request.destination: 0}
        queue = deque([request.destination])
        while queue:
            node = queue.popleft()
            for neighbor in env.resources.snapshot.neighbors(node):
                if neighbor not in distances:
                    distances[neighbor] = distances[node] + 1
                    queue.append(neighbor)
        actions = [i for i, legal in enumerate(observation["feasible_action_mask"]) if legal]
        action = min(actions, key=lambda i: (distances.get(observation["node_ids"][i], 999), i)) if actions else 0
        next_observation, reward, terminated, truncated, info = env.step(action) if actions else env.finish_without_action()
        trajectory.append({"state": observation, "action": action, "reward": reward,
                           "next_state": next_observation, "terminated": terminated,
                           "truncated": truncated, "info": info})
        observation = next_observation
        done = terminated or truncated
    before_expiry = env.resources.metrics()
    env.advance_time(request.expires_s)
    return {"mode": "mdp_fixture", "policy": "shortest_hop_fixture_not_trained_gahso",
            "trajectory": trajectory, "before_expiry": before_expiry,
            "after_expiry": env.resources.metrics(), "events": env.resources.events}


"""以真實 25 顆衛星 MDP 狀態執行 GAT 前向傳播與反向傳播。"""
from dataclasses import asdict
import torch
from gahso.features import GATFeatureBuilder
from gahso.gat import GAHSOGAT
from gahso.mdp import SFCOrchestrationEnv
from gahso.network import LEOConstellation
from gahso.services import SFCRequest, VNF


def check_gat_gradients(config) -> dict:
    """隨機初始化模型的介面驗證；測試 loss 不代表 GAHSO 訓練目標。"""
    env = SFCOrchestrationEnv(LEOConstellation(config.constellation), config.mdp,
                             cpu_load_weight=config.cpu_load_weight,
                             bandwidth_load_weight=config.bandwidth_load_weight)
    request = SFCRequest("gat-fixture", 0, 2,
                         (VNF("ingress", 50), VNF("process", 150), VNF("egress", 75)),
                         30, 0.1, 0, 30)
    observation, _ = env.reset(request=request)
    builder = GATFeatureBuilder(config.gat_features)
    graph = builder.build(observation)
    # 僅在示範中固定初始化種子，不改動外部訓練程序的全域 RNG 狀態。
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(7)
        model = GAHSOGAT(config.gat_features, config.gat)
        embeddings, attentions = model(graph, return_attention=True)
        loss = embeddings.square().mean()
        loss.backward()
    gradients = {
        name: {"finite": p.grad is not None and bool(torch.isfinite(p.grad).all()),
               "norm": float(p.grad.norm()) if p.grad is not None else None}
        for name, p in model.named_parameters()
    }
    attention_error = []
    for weights in attentions:
        sums = weights.alpha.new_zeros((graph.x.shape[0], config.gat.heads))
        sums.index_add_(0, weights.edge_index[1], weights.alpha.detach())
        attention_error.append(float((sums - 1).abs().max()))
    return {
        "mode": "gat_fixture", "torch_version": torch.__version__,
        "device": str(graph.x.device), "trained": False,
        "feature_config": asdict(config.gat_features), "gat_config": asdict(config.gat),
        "node_input_shape": list(graph.x.shape), "edge_input_shape": list(graph.edge_attr.shape),
        "edge_index_shape": list(graph.edge_index.shape), "output_shape": list(embeddings.shape),
        "node_feature_names": graph.node_feature_names,
        "edge_feature_names": graph.edge_feature_names,
        "attention_normalization_max_error": attention_error,
        "backward_check_loss": float(loss.detach()),
        "gradients": gradients, "embeddings": embeddings.detach().tolist(),
        "attention": [{"edge_index": a.edge_index.tolist(), "alpha": a.alpha.detach().tolist()}
                      for a in attentions],
        "resources_unchanged": env.resources.metrics()["cpu_reserved"] == 0
                               and env.resources.metrics()["bandwidth_reserved_mhz"] == 0,
    }


"""小型訓練示範：每個請求完成後實際更新一次，非收斂／效能評估。"""
import torch
from gahso.actor_critic import ActorCriticAgent
from gahso.mdp import SFCOrchestrationEnv
from gahso.network import LEOConstellation
from gahso.services import SFCRequest, VNF
from gahso.training import A2CTrainer


def check_training_updates(config):
    env = SFCOrchestrationEnv(LEOConstellation(config.constellation), config.mdp,
                             cpu_load_weight=config.cpu_load_weight,
                             bandwidth_load_weight=config.bandwidth_load_weight)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(7)
        agent = ActorCriticAgent(config.constellation.satellites, config.gat_features,
                                config.gat, config.agent)
    trainer = A2CTrainer(env, agent, config.training)
    reports = []
    for index in range(3):
        q = SFCRequest(f"training-{index}", 0, 2,
                       (VNF("in", 50), VNF("work", 150), VNF("out", 75)),
                       30, .1, index, 30)
        reports.append(trainer.train_request(q, include_trajectory=True))
    before = env.resources.metrics()
    env.advance_time(32)
    return {"mode": "train_fixture", "requests_updated": trainer.requests_updated,
            "actor_updates": trainer.actor_updates, "critic_updates": int(agent.critic_updates),
            "accepted_requests": sum(r["accepted"] for r in reports),
            "total_reward": sum(r["total_reward"] for r in reports),
            "requests": reports, "before_expiry": before, "after_expiry": env.resources.metrics()}
