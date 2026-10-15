"""GAHSO 第 VI-A 節：逐跳 SFC 編排的 MDP（不依賴 Gymnasium 或神經網路）。

reset/start_request 回傳 (observation, info)；
step 回傳 (observation, reward, terminated, truncated, info)。
這是五回傳值介面，不宣稱實作 gymnasium.Env / spaces。
一個服務的 K 步內固定網路快照；只有服務之間推進物理時間。
"""
from collections import deque
from dataclasses import dataclass, field, asdict, replace
from math import isfinite
from .embedding import embed_vnfs
from .network import LEOConstellation, edge_key
from .resources import ResourceManager, jain_fairness, EPSILON
from .services import SFCRequest
from .routing import remaining_delays, arrival_diagnostics, failure_category


@dataclass(frozen=True)
class RewardConfig:
    """公式 (17)、(18) 的係數；下列數值為可調工程預設，非論文公布值。

    legacy 保留以秒和 CPU/MHz 加權值計算的相容預設。
    正式 JSON 設定選用 normalized 與較小係數，依請求上限及容量計算比例。
    """
    theta1: float = 1000.0
    theta2: float = 1.0
    theta3: float = 0.01
    theta4: float = 1000.0
    theta5: float = 1.0
    theta6: float = 1.0
    alpha1: float = 10.0
    alpha2: float = 20.0
    beta0: float = 20.0
    beta1: float = 5.0
    gamma1: float = 1.0
    mode: str = "legacy"

    def __post_init__(self):
        if self.mode not in ("legacy", "normalized"):
            raise ValueError("Unknown reward mode")
        if any(not isfinite(v) or v < 0 for k, v in asdict(self).items() if k != "mode"):
            raise ValueError("Reward coefficients must be finite and nonnegative")
        if self.beta0 <= self.beta1:
            raise ValueError("beta0 must exceed beta1")


@dataclass(frozen=True)
class MDPConfig:
    max_steps: int = 25  # Table I 的 K 範圍是 20–30。
    load_scale: float = 1000.0  # 僅特徵正規化；不改動公式中的原始負載。
    delay_scale_s: float = 0.1
    rewards: RewardConfig = field(default_factory=RewardConfig)
    delay_feasibility_mask: bool = True
    endpoint_placement_mode: str = "fixed"

    def __post_init__(self):
        if type(self.max_steps) is not int or self.max_steps <= 0:
            raise ValueError("max_steps must be a positive integer")
        if not all(isfinite(x) and x > 0 for x in (self.load_scale, self.delay_scale_s)):
            raise ValueError("Feature scales must be finite and positive")
        if self.endpoint_placement_mode not in ("fixed", "neighbor_allowed"):
            raise ValueError("endpoint_placement_mode must be fixed or neighbor_allowed")

    @classmethod
    def from_dict(cls, values: dict):
        values = dict(values)
        values["rewards"] = RewardConfig(**values.get("rewards", {}))
        return cls(**values)


class SFCOrchestrationEnv:
    """每個 episode 編排一個 SFC；start_request 保留其他已接受服務的資源。

    step 的整數動作是 node_ids 的索引，可由 policy 的 one-hot 動作 argmax 得到。
    25 顆衛星的預設 node_ids 為 0..24，索引恰好就是衛星 ID。
    """
    NODE_FEATURE_NAMES = (
        "cpu_remaining_ratio", "load_scaled", "is_source", "is_destination",
        "visited", "is_current", "remaining_delay_bound_scaled", "destination_reachable",
    )
    EDGE_FEATURE_NAMES = ("bandwidth_remaining_ratio", "delay_scaled")

    def __init__(self, constellation: LEOConstellation | None = None,
                 config: MDPConfig | None = None, *,
                 cpu_load_weight: float = 1.0, bandwidth_load_weight: float = 1.0):
        self.constellation = constellation or LEOConstellation()
        self.config = config or MDPConfig()
        self.cpu_load_weight = cpu_load_weight
        self.bandwidth_load_weight = bandwidth_load_weight
        self.reset()

    def reset(self, *, request: SFCRequest | None = None, seed: int | None = None,
              time_s: float = 0.0):
        """清空整個網路帳本，開始新訓練回合；可同時指定第一個請求。"""
        if seed is not None:
            self.constellation = LEOConstellation(replace(self.constellation.config, seed=seed))
        self.resources = ResourceManager(self.constellation.snapshot(time_s),
                                         self.cpu_load_weight, self.bandwidth_load_weight)
        self.request = None
        self.path = []
        self.steps = 0
        self.delay_s = 0.0
        self._done = True
        self._accepted = False
        self._terminated = False
        self._hops = {}
        self._allocation = None
        self._delay_cache = None
        self._initial_diagnostics = None
        if request is not None:
            return self.start_request(request)
        return self.observation(), {"ready": True}

    def advance_time(self, time_s: float):
        """服務邊界更新拓樸、釋放到期服務；逐跳決策中禁止改變快照。"""
        if not self._done:
            raise RuntimeError("Finish the current request before advancing physical time")
        self.resources.update(self.constellation.snapshot(time_s))

    def start_request(self, request: SFCRequest):
        """保留背景流量，啟動另一請求；不隱含 reset 或重複配置。"""
        if not self._done:
            raise RuntimeError("The current request has not finished")
        if request.id in self.resources.records:
            raise ValueError("Duplicate request ID")
        if request.source not in self.resources.snapshot.satellites or request.destination not in self.resources.snapshot.satellites:
            raise ValueError("Unknown request endpoint")
        next_time = max(self.resources.time_s, request.arrival_s)
        if next_time >= request.expires_s:
            raise ValueError("Request has already expired")
        self.advance_time(next_time)
        self.request = request
        self.path = [request.source]
        self.steps, self.delay_s = 0, 0.0
        self._done, self._accepted, self._terminated = False, False, False
        self._allocation = None
        self._delay_cache = None
        self._initial_diagnostics = arrival_diagnostics(
            self.resources, request, self.config.endpoint_placement_mode)
        # 無權圖 BFS 一次算好到目的地的最短跳數，供每一步公式 (17) 使用。
        self._hops = {request.destination: 0}
        queue = deque([request.destination])
        while queue:
            node = queue.popleft()
            for neighbor in self.resources.snapshot.neighbors(node):
                if neighbor not in self._hops:
                    self._hops[neighbor] = self._hops[node] + 1
                    queue.append(neighbor)
        return self.observation(), {"request_id": request.id, "ready": True}

    @property
    def node_ids(self):
        return tuple(sorted(self.resources.snapshot.satellites))

    @property
    def current_node(self):
        return self.path[-1] if self.path else None

    def _hop_distance(self, node):
        # 不連通時使用有限哨兵 N；避免 inf 獎勵污染梯度。
        return self._hops.get(node, len(self.node_ids))

    def _remaining_delays(self):
        key = tuple(self.path)
        if self._delay_cache is None or self._delay_cache[0] != key:
            self._delay_cache = (key, remaining_delays(self.resources, self.request.destination,
                                                       self.request.bandwidth_mhz, self.path))
        return self._delay_cache[1]

    def action_reasons(self, action: int) -> list[str]:
        """All applicable rejection reasons. A path lower bound is not a CPU proof."""
        if self._done:
            return ["episode_not_active"]
        if type(action) is not int or not 0 <= action < len(self.node_ids):
            return ["invalid_action_index"]
        node = self.node_ids[action]
        if len(self.path) == 1 and self.request.source == self.request.destination and node == self.current_node:
            return []
        if node not in self.resources.snapshot.neighbors(self.current_node):
            return ["not_neighbor"]
        reasons = []
        if node in self.path:
            reasons.append("path_contains_cycle")
        if self.resources.bandwidth_available(self.current_node, node) + EPSILON < self.request.bandwidth_mhz:
            reasons.append("insufficient_bandwidth")
        if self.config.delay_feasibility_mask and node not in self.path:
            bound = self._remaining_delays().get(node)
            if bound is None:
                reasons.append("destination_unreachable")
            else:
                hop = self.resources.snapshot.links[edge_key(self.current_node, node)].delay_s
                if self.delay_s + hop + bound > self.request.max_delay_s + 1e-12:
                    reasons.append("delay_lower_bound_exceeded")
        return reasons

    def check_action(self, action: int) -> str:
        reasons = self.action_reasons(action)
        return reasons[0] if reasons else ""


    def _tentative_resources(self):
        """路徑頻寬在 observation 中暫扣，帳本直到成功終止才正式提交。"""
        manager = self.resources
        bandwidth = {edge: manager.bandwidth_available(*edge) for edge in manager.snapshot.links}
        loads = manager.node_loads()
        if self.request is not None and not self._done:
            for a, b in zip(self.path, self.path[1:]):
                bandwidth[edge_key(a, b)] -= self.request.bandwidth_mhz
                extra = manager.bandwidth_weight * self.request.bandwidth_mhz
                loads[a] += extra
                loads[b] += extra
        return bandwidth, loads

    def observation(self) -> dict:
        """s_t={G_t,C_t,B_t,l_t,I_t(q)}，並提供可直接轉 tensor 的圖特徵。

        raw 資源量與正規化特徵並存；step_count/path 使有限步數與循環限制可觀測。
        CPU 尚未正式嵌入前不暫扣；終止失敗的候選頻寬會從 observation 回復。
        """
        manager = self.resources
        ids = self.node_ids
        index = {node: i for i, node in enumerate(ids)}
        bandwidth, loads = self._tentative_resources()
        req = self.request
        source = req.source if req else None
        destination = req.destination if req else None
        bounds = self._remaining_delays() if req and not self._done else {}
        cpu = tuple(manager.cpu_available(n) for n in ids)
        node_features = tuple((
            manager.cpu_available(n) / manager.snapshot.satellites[n].cpu_capacity,
            loads[n] / self.config.load_scale,
            float(n == source), float(n == destination), float(n in self.path),
            float(n == self.current_node),
            min(bounds[n] / req.max_delay_s, 5.0) if n in bounds else 0.0,
            float(n in bounds),
        ) for n in ids)
        # 無向 ISL 轉為雙向訊息邊；共享同一筆實體頻寬，不能重複扣款。
        left, right, edge_features = [], [], []
        for (a, b), link in sorted(manager.snapshot.links.items()):
            for u, v in ((a, b), (b, a)):
                left.append(index[u])
                right.append(index[v])
                edge_features.append((bandwidth[(a, b)] / link.capacity_mhz,
                                      link.delay_s / self.config.delay_scale_s))
        neighbor_ids = set(manager.snapshot.neighbors(self.current_node)) if self.path else set()
        if req and source == destination and len(self.path) == 1:
            neighbor_ids = {source}
        return {
            "time_s": manager.time_s,
            "node_ids": ids,
            "node_features": node_features,
            "node_feature_names": self.NODE_FEATURE_NAMES,
            "edge_index": (tuple(left), tuple(right)),
            "edge_features": tuple(edge_features),
            "edge_feature_names": self.EDGE_FEATURE_NAMES,
            "cpu_remaining": cpu,
            "link_remaining": tuple((a, b, value) for (a, b), value in sorted(bandwidth.items())),
            "node_loads": tuple(loads[n] for n in ids),
            "fairness": jain_fairness(loads.values()),
            "request": asdict(req) if req else None,
            "path": tuple(self.path),
            "position": tuple(int(n == self.current_node) for n in ids),
            "step_count": self.steps,
            "max_steps": self.config.max_steps,
            "steps_remaining": max(0, self.config.max_steps - self.steps),
            "accumulated_delay_s": self.delay_s,
            # 論文只遮罩非鄰居：回訪／頻寬不足仍可選，交由非法動作懲罰處理。
            "action_mask": tuple(not self._done and n in neighbor_ids for n in ids),
            "feasible_action_mask": tuple(not self._done and not self.check_action(i) for i in range(len(ids))),
        }

    def _finish(self, outcome: str, reason: str, *, unembedded: int = 0,
                fairness: float | None = None, feasibility=None, embedding=None):
        """公式 (18)：終止獎勵取代當步即時獎勵，不重複相加。"""
        weights = self.config.rewards
        fairness = self.resources.fairness() if fairness is None else fairness
        if weights.mode == "normalized":
            # 固定相對尺度：成功路徑的逐跳延遲成本總和 <= theta1。
            # 終止延遲成本再以整條路徑計算，避免漏掉最後一跳。
            delay_ratio = min(self.delay_s / self.request.max_delay_s, 5.0)
            components = {"success" if outcome == "success" else "failure":
                          weights.beta0 if outcome == "success" else -weights.alpha2,
                          "fairness": weights.gamma1 * fairness}
            if outcome in ("success", "delay_violation"):
                components["delay"] = -weights.theta4 * delay_ratio
            else:
                components["hops"] = -weights.theta5 * self._hop_distance(self.current_node) / len(self.node_ids)
                components["unembedded"] = -weights.theta6 * unembedded / len(self.request.vnfs)
        elif outcome == "success":
            components = {"success": weights.beta0, "fairness": weights.gamma1 * fairness}
        elif outcome == "delay_violation":
            components = {"delay_base": weights.beta1, "delay": -weights.theta4 * self.delay_s,
                          "fairness": weights.gamma1 * fairness}
        else:
            components = {"failure": -weights.alpha2,
                          "hops": -weights.theta5 * self._hop_distance(self.current_node),
                          "unembedded": -weights.theta6 * unembedded,
                          "fairness": weights.gamma1 * fairness}
        reward = sum(components.values())
        neighbors = self.resources.snapshot.neighbors(self.current_node)
        blocked = {str(n): self.action_reasons(self.node_ids.index(n)) for n in neighbors}
        blocked = {n: reasons for n, reasons in blocked.items() if reasons}
        no_action_reasons = sorted({r for reasons in blocked.values() for r in reasons})
        if not neighbors:
            no_action_reasons.append("no_physical_neighbors")
        self._done = True
        # K 是論文任務內的失敗條件，故 terminated=True；不是外加時間截斷。
        self._terminated = True
        if not self._accepted:
            self.resources.reject(self.request, reason)
        self.resources.assert_invariants()
        info = {
            "request_id": self.request.id, "outcome": outcome, "reason": reason,
            "initial_diagnostics": self._initial_diagnostics,
            "failure_category": failure_category(self._initial_diagnostics, reason, self._accepted),
            "blocked_actions": blocked, "no_action_reasons": no_action_reasons,
            "accepted": self._accepted, "unembedded_vnfs": unembedded,
            "fairness_for_reward": fairness, "terminated": True,
            "reward_components": components,
            "step_limit_reached": self.steps >= self.config.max_steps,
            "allocation": asdict(self._allocation) if self._allocation else None,
            "feasibility": asdict(feasibility) if feasibility else None,
            # 候選放置也記錄在失敗報告；allocation 仍只表示已成功提交。
            "embedding": asdict(embedding) if embedding is not None else None,
        }
        return self.observation(), reward, True, False, info

    def _finish_at_destination(self):
        """Algorithm 2 放置後檢查全部約束，只有真正可行時才 admit。"""
        embedding = embed_vnfs(self.resources, self.request, tuple(self.path),
                               self.config.endpoint_placement_mode)
        if embedding.unembedded:
            return self._finish("failure", "vnf_embedding_failed", unembedded=embedding.unembedded, embedding=embedding)
        allocation = embedding.allocation
        report = self.resources.check_allocation(self.request, allocation)
        if report.feasible:
            decision = self.resources.admit(self.request, allocation)
            if not decision.accepted:
                return self._finish("failure", decision.reason, embedding=embedding)
            self._accepted, self._allocation = True, allocation
            return self._finish("success", "accepted", feasibility=report, embedding=embedding)
        if all(check.passed or check.code == "C4" for check in report.checks):
            # 超時配置只做反事實資源預覽；正式帳本維持不變。
            fairness = jain_fairness(self.resources.projected_loads(self.request, allocation).values())
            return self._finish("delay_violation", "delay_exceeded", fairness=fairness, feasibility=report, embedding=embedding)
        return self._finish("failure", report.reason, feasibility=report, embedding=embedding)

    def finish_without_action(self):
        """無任何鄰居可選時結束請求，避免對全遮罩 Actor 建立無效分布。

        這是沒有 policy action 的環境終止事件，不增加決策步數；
        後續訓練可以更新 Critic，但不可捏造 log_prob 來更新 Actor。
        """
        if self._done:
            raise RuntimeError("No active request")
        if any(self.observation()["feasible_action_mask"]):
            raise ValueError("Feasible actions remain; select an action with step()")
        embedding = embed_vnfs(self.resources, self.request, tuple(self.path),
                               self.config.endpoint_placement_mode)
        return self._finish("failure", "dead_end", unembedded=embedding.unembedded, embedding=embedding)

    def step(self, action: int):
        """選下一跳；非法動作留在原位置但消耗一步，避免無限重試。"""
        if self._done:
            raise RuntimeError("Call start_request() or reset(request=...) before step()")
        reason = self.check_action(action)
        self.steps += 1
        if reason:
            components = {"invalid_action": -self.config.rewards.alpha1}
            instant_reward = sum(components.values())
        else:
            node = self.node_ids[action]
            hop_delay = 0.0
            if node != self.current_node:
                hop_delay = self.resources.snapshot.links[edge_key(self.current_node, node)].delay_s
                self.path.append(node)
                self.delay_s += hop_delay
            if node == self.request.destination:
                return self._finish_at_destination()
            _, loads = self._tentative_resources()
            w = self.config.rewards
            components = {"delay": -w.theta1 * hop_delay,
                          "hops": -w.theta2 * self._hop_distance(node),
                          "load": -w.theta3 * loads[node]}
            if w.mode == "normalized":
                # 容量加權的負載比例；總 hop/load 成本由 max_steps 限定。
                capacity = (self.cpu_load_weight * self.resources.snapshot.satellites[node].cpu_capacity
                            + self.bandwidth_load_weight * sum(link.capacity_mhz
                              for edge, link in self.resources.snapshot.links.items() if node in edge))
                components = {
                    "delay": -w.theta1 * min(hop_delay / self.request.max_delay_s, 1.0),
                    "hops": -w.theta2 * self._hop_distance(node) / len(self.node_ids) / self.config.max_steps,
                    "load": -w.theta3 * min(loads[node] / max(capacity, EPSILON), 1.0) / self.config.max_steps,
                }
            instant_reward = sum(components.values())
        # K 邊界仍允許該步成功到達；否則以公式 (18) 第三分支結束。
        dead_end = not any(not self.check_action(i) for i in range(len(self.node_ids)))
        if self.steps >= self.config.max_steps or dead_end:
            embedding = embed_vnfs(self.resources, self.request, tuple(self.path),
                                   self.config.endpoint_placement_mode)
            result = self._finish("failure", "max_steps" if self.steps >= self.config.max_steps else "dead_end",
                                  unembedded=embedding.unembedded, embedding=embedding)
            result[-1]["last_action_reason"] = reason
            result[-1]["instant_reward_before_terminal"] = instant_reward
            result[-1]["instant_reward_components"] = components
            return result
        return self.observation(), instant_reward, False, False, {
            "request_id": self.request.id, "valid_action": not reason, "reason": reason,
            "accepted": False, "outcome": "in_progress", "reward_components": components,
        }
