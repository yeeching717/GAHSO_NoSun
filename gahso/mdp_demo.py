"""可執行的 MDP 介面示範；以最短跳數選動作，不宣稱為訓練後的 GAHSO。"""
from collections import deque
from .mdp import SFCOrchestrationEnv
from .network import LEOConstellation
from .services import SFCRequest, VNF


def run_mdp_demo(config) -> dict:
    """一個小型請求完整走過 reset→step→接受/拒絕→到期釋放。"""
    env = SFCOrchestrationEnv(LEOConstellation(config.constellation), config.mdp,
                             cpu_load_weight=config.cpu_load_weight,
                             bandwidth_load_weight=config.bandwidth_load_weight)
    request = SFCRequest("mdp-demo", 0, 2, tuple(VNF(f"vnf-{i}", 75) for i in range(3)),
                         30, 0.1, 0, 30, "custom")
    observation, _ = env.reset(request=request)
    trajectory = []
    done = False
    while not done:
        # 僅供示範：BFS 距離作為啟發式分數；未來換成 Actor 的 logits 即可。
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
        next_observation, reward, terminated, truncated, info = env.step(action)
        trajectory.append({"state": observation, "action": action, "reward": reward,
                           "next_state": next_observation, "terminated": terminated,
                           "truncated": truncated, "info": info})
        observation = next_observation
        done = terminated or truncated
    before_expiry = env.resources.metrics()
    env.advance_time(request.expires_s)
    return {"mode": "mdp_demo", "policy": "shortest_hop_demo_not_trained_gahso",
            "trajectory": trajectory, "before_expiry": before_expiry,
            "after_expiry": env.resources.metrics(), "events": env.resources.events}
