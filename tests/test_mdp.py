"""MDP 使用可手算的四節點拓樸，避免測試結果依賴衛星幾何。"""
import math
import unittest
from dataclasses import replace
from gahso.network import ConstellationConfig, Link, NetworkSnapshot, Satellite, edge_key
from gahso.resources import Allocation, ResourceManager, jain_fairness
from gahso.services import SFCRequest, VNF
from gahso.mdp import MDPConfig, RewardConfig, SFCOrchestrationEnv
from gahso.embedding import embed_vnfs
from gahso.simulation import SimulationConfig
from gahso.mdp_demo import run_mdp_demo


class FixedConstellation:
    """只有時間變化的手工圖，供決策與獎勵公式測試。"""
    config = ConstellationConfig()

    def __init__(self, edges=((0, 1), (1, 2), (2, 3)), capacity=200, cpu=500):
        self.edges, self.capacity, self.cpu = edges, capacity, cpu

    def snapshot(self, time_s):
        nodes = {n: Satellite(n, 0, n, (8000, n * 10, 0), self.cpu) for n in range(4)}
        links = {edge_key(a, b): Link(*edge_key(a, b), 2990, 0.01, self.capacity, "test")
                 for a, b in self.edges}
        return NetworkSnapshot(time_s, nodes, links)


def request(identifier="q", cpu=100, delay=0.1, arrival=0, duration=10):
    return SFCRequest(identifier, 0, 3, tuple(VNF(str(i), cpu) for i in range(3)),
                      50, delay, arrival, duration, "compute")


def environment(**kwargs):
    return SFCOrchestrationEnv(FixedConstellation(), MDPConfig(**kwargs))


class FeasibilityTests(unittest.TestCase):
    def test_all_constraints_pass_and_check_is_read_only(self):
        manager = ResourceManager(FixedConstellation().snapshot(0))
        plan = Allocation((0, 1, 2, 3), (0, 1, 3))
        report = manager.check_allocation(request(), plan)
        self.assertTrue(report.feasible)
        for code in ("C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8"):
            self.assertTrue(next(c.passed for c in report.checks if c.code == code))
        self.assertAlmostEqual(report.delay_s, 0.03)
        self.assertEqual(manager.metrics()["cpu_reserved"], 0)
        self.assertEqual(manager.records, {})

    def test_multiple_violations_and_flow_conservation(self):
        manager = ResourceManager(FixedConstellation().snapshot(0))
        q = replace(request(cpu=600, delay=0.001), bandwidth_mhz=250)
        report = manager.check_allocation(q, Allocation((0, 1, 2, 3), (0, 1, 3)))
        self.assertEqual(set(report.violations), {"delay_exceeded", "insufficient_cpu", "insufficient_bandwidth"})
        q = replace(request(), vnfs=tuple(VNF(str(i), 100) for i in range(4)))
        report = manager.check_allocation(q, Allocation((0, 1, 2, 3), (0, 2, 1, 3)))
        self.assertIn("flow_conservation_violation", report.violations)

    def test_binary_representation_rejects_fractional_node_ids(self):
        with self.assertRaises(ValueError):
            Allocation((0, 1.5, 3), (0, 1, 3))

    def test_weighted_load_and_fairness(self):
        manager = ResourceManager(FixedConstellation().snapshot(0), cpu_weight=2, bandwidth_weight=3)
        manager.admit(request(), Allocation((0, 1, 2, 3), (0, 1, 3)))
        self.assertEqual(manager.node_loads(), {0: 350, 1: 500, 2: 300, 3: 350})
        self.assertAlmostEqual(manager.fairness(), 1500 ** 2 / (4 * (350 ** 2 * 2 + 500 ** 2 + 300 ** 2)))

    def test_projected_load_includes_cpu_and_bandwidth_without_mutation(self):
        manager = ResourceManager(FixedConstellation().snapshot(0))
        loads = manager.projected_loads(request(), Allocation((0, 1, 2, 3), (0, 1, 3)))
        self.assertEqual(loads, {0: 150, 1: 200, 2: 100, 3: 150})
        self.assertEqual(manager.node_loads(), {0: 0, 1: 0, 2: 0, 3: 0})
        self.assertEqual(manager.events, [])

    def test_jain_boundaries_and_numerical_stability(self):
        self.assertEqual(jain_fairness([0, 0, 0, 0]), 1)
        self.assertEqual(jain_fairness([2, 2, 2, 2]), 1)
        self.assertEqual(jain_fairness([2, 0, 0, 0]), 0.25)
        self.assertEqual(jain_fairness([1e200, 1e200]), 1)
        for values in ([], [-1, 1], [float("nan")], [float("inf")]):
            with self.subTest(values=values), self.assertRaises(ValueError):
                jain_fairness(values)

    def test_embedding_keeps_order_and_counts_missing_vnfs(self):
        manager = ResourceManager(FixedConstellation().snapshot(0))
        partial = embed_vnfs(manager, request(), (0, 1))
        self.assertEqual(partial.placements, (0, 1, None))
        self.assertEqual(partial.unembedded, 1)
        self.assertIsNone(partial.allocation)
        full = embed_vnfs(manager, request(), (0, 1, 2, 3))
        self.assertEqual(full.allocation.vnf_nodes, (0, 1, 3))
        self.assertEqual(manager.metrics()["cpu_reserved"], 0)


class MDPTests(unittest.TestCase):
    def test_observation_has_markov_state_graph_and_masks(self):
        env = environment()
        state, info = env.reset(request=request())
        self.assertEqual(state["node_ids"], (0, 1, 2, 3))
        self.assertEqual(len(state["node_features"]), 4)
        self.assertEqual(len(state["node_features"][0]), len(env.NODE_FEATURE_NAMES))
        self.assertEqual(len(state["edge_features"]), 6)
        self.assertEqual(len(state["edge_index"]), 2)
        self.assertEqual(state["action_mask"], (False, True, False, False))
        self.assertEqual(state["position"], (1, 0, 0, 0))
        self.assertEqual(state["path"], (0,))
        self.assertEqual(state["request"]["vnfs"][0]["cpu"], 100)
        self.assertEqual(state["steps_remaining"], 25)

    def test_instant_reward_matches_equation_17_and_tentative_resources(self):
        env = environment()
        env.reset(request=request())
        state, reward, terminal, truncated, info = env.step(1)
        # d=0.01 秒、距目的地 2 跳、下一節點暫存通訊負載 50。
        self.assertAlmostEqual(reward, -1000 * 0.01 - 2 - 0.01 * 50)
        self.assertFalse(terminal or truncated)
        self.assertTrue(info["valid_action"])
        self.assertEqual(state["link_remaining"][0], (0, 1, 150))
        self.assertEqual(state["node_loads"], (50, 50, 0, 0))
        self.assertEqual(state["steps_remaining"], 24)
        self.assertEqual(env.resources.bandwidth_available(0, 1), 200)
        self.assertEqual(env.resources.metrics()["cpu_reserved"], 0)

    def test_success_reward_commit_and_expiry(self):
        env = environment()
        env.reset(request=request())
        env.step(1)
        env.step(2)
        state, reward, terminal, truncated, info = env.step(3)
        self.assertTrue(terminal)
        self.assertFalse(truncated)
        self.assertEqual(info["outcome"], "success")
        self.assertTrue(info["accepted"])
        self.assertAlmostEqual(reward, 20 + env.resources.fairness())
        self.assertEqual(env.resources.metrics()["cpu_reserved"], 300)
        self.assertEqual(env.resources.metrics()["bandwidth_reserved_mhz"], 150)
        self.assertEqual(state["action_mask"], (False,) * 4)
        with self.assertRaises(RuntimeError):
            env.step(3)
        env.advance_time(10)
        self.assertEqual(env.resources.metrics()["cpu_reserved"], 0)
        self.assertEqual(env.resources.metrics()["bandwidth_reserved_mhz"], 0)

    def test_latency_branch_uses_projected_fairness_without_commit(self):
        env = environment()
        q = request(delay=0.015)
        env.reset(request=q)
        projected = env.resources.projected_loads(q, Allocation((0, 1, 2, 3), (0, 1, 3)))
        env.step(1)
        env.step(2)  # 尚未到達，超時不提早結束，讓公式 (18) 第二分支可達。
        state, reward, terminal, _, info = env.step(3)
        self.assertEqual(info["outcome"], "delay_violation")
        self.assertFalse(info["accepted"])
        self.assertAlmostEqual(reward, 5 - 1000 * 0.03 + jain_fairness(projected.values()))
        self.assertEqual(env.resources.metrics()["cpu_reserved"], 0)
        self.assertEqual(env.resources.metrics()["bandwidth_reserved_mhz"], 0)
        self.assertEqual(state["node_loads"], (0, 0, 0, 0))

    def test_cpu_embedding_failure_reports_missing_count(self):
        env = environment()
        env.reset(request=request(cpu=600))
        env.step(1)
        env.step(2)
        _, reward, terminal, _, info = env.step(3)
        self.assertTrue(terminal)
        self.assertEqual(info["unembedded_vnfs"], 3)
        self.assertEqual(reward, -20 - 3 + 1)
        self.assertEqual(env.resources.metrics()["cpu_reserved"], 0)

    def test_invalid_action_stays_in_place_and_consumes_one_step(self):
        env = environment()
        env.reset(request=request())
        for action in (3, -1, [0, 1, 0, 0]):
            state, reward, terminal, _, info = env.step(action)
            self.assertEqual(state["path"], (0,))
            self.assertEqual(reward, -10)
            self.assertFalse(terminal)
            self.assertFalse(info["valid_action"])
        self.assertEqual(state["step_count"], 3)

    def test_mask_is_neighbors_only_and_cycle_is_penalized(self):
        env = environment()
        env.reset(request=request())
        state, *_ = env.step(1)
        self.assertTrue(state["action_mask"][0])
        self.assertFalse(state["feasible_action_mask"][0])
        state, reward, _, _, info = env.step(0)
        self.assertEqual(state["path"], (0, 1))
        self.assertEqual(info["reason"], "path_contains_cycle")
        self.assertEqual(reward, -10)

    def test_capacity_invalid_action_and_dead_end_are_finite(self):
        env = SFCOrchestrationEnv(FixedConstellation(capacity=20))
        state, _ = env.reset(request=request())
        self.assertTrue(state["action_mask"][1])
        self.assertFalse(state["feasible_action_mask"][1])
        state, reward, terminated, truncated, info = env.step(1)
        self.assertTrue(terminated)
        self.assertFalse(truncated)
        self.assertEqual(info["last_action_reason"], "insufficient_bandwidth")
        self.assertEqual(info["reason"], "dead_end")
        self.assertTrue(math.isfinite(reward))
        self.assertEqual(env.resources.metrics()["bandwidth_reserved_mhz"], 0)

    def test_max_steps_is_task_termination_and_counts_unembedded(self):
        env = environment(max_steps=1)
        env.reset(request=request())
        _, reward, terminal, truncated, info = env.step(1)
        self.assertEqual(info["reason"], "max_steps")
        self.assertEqual(info["unembedded_vnfs"], 1)
        self.assertEqual(reward, -20 - 2 - 1 + 1)
        self.assertTrue(terminal)
        self.assertFalse(truncated)

    def test_success_allowed_exactly_at_step_limit(self):
        env = environment(max_steps=3)
        env.reset(request=request())
        env.step(1)
        env.step(2)
        self.assertTrue(env.step(3)[-1]["accepted"])

    def test_disconnected_and_isolated_nodes_do_not_produce_nan(self):
        for edges in ((), ((0, 1),)):
            with self.subTest(edges=edges):
                env = SFCOrchestrationEnv(FixedConstellation(edges=edges))
                env.reset(request=request())
                _, reward, done, _, info = env.step(1 if edges else 0)
                self.assertTrue(done)
                self.assertTrue(math.isfinite(reward))
                self.assertEqual(info["reason"], "dead_end")

    def test_start_request_preserves_background_and_reset_clears_it(self):
        env = environment()
        env.reset(request=request())
        for node in (1, 2, 3):
            env.step(node)
        state, _ = env.start_request(request("second", arrival=1))
        self.assertEqual(env.resources.metrics()["cpu_reserved"], 300)
        self.assertEqual(state["time_s"], 1)
        self.assertEqual(state["link_remaining"][0], (0, 1, 150))
        with self.assertRaises(RuntimeError):
            env.advance_time(2)
        with self.assertRaises(RuntimeError):
            env.start_request(request("third"))
        env.reset()
        self.assertEqual(env.resources.records, {})

    def test_expiry_at_next_request_boundary(self):
        env = environment()
        env.reset(request=request(duration=1))
        for node in (1, 2, 3):
            env.step(node)
        env.start_request(request("second", arrival=1))
        self.assertEqual(env.resources.metrics()["cpu_reserved"], 0)

    def test_duplicate_unknown_and_expired_requests(self):
        env = environment()
        with self.assertRaises(ValueError):
            env.start_request(replace(request(), destination=8))
        env.advance_time(10)
        with self.assertRaises(ValueError):
            env.start_request(request())
        env.reset(request=request())
        for node in (1, 2, 3):
            env.step(node)
        with self.assertRaises(ValueError):
            env.start_request(request())

    def test_same_source_destination_local_completion(self):
        env = environment()
        q = replace(request(), destination=0)
        state, _ = env.reset(request=q)
        self.assertEqual(state["action_mask"], (True, False, False, False))
        _, reward, done, _, info = env.step(0)
        self.assertTrue(info["accepted"])
        self.assertEqual(env.resources.metrics()["bandwidth_reserved_mhz"], 0)
        self.assertEqual(env.resources.metrics()["cpu_reserved"], 300)

    def test_seeded_reset_is_reproducible(self):
        env = SFCOrchestrationEnv()
        first, _ = env.reset(seed=42)
        second, _ = env.reset(seed=42)
        self.assertEqual(first, second)
        self.assertEqual(len(first["node_ids"]), 20)

    def test_observation_does_not_alias_internal_state(self):
        env = environment()
        state, _ = env.reset(request=request())
        state["request"]["source"] = 999
        state["request"]["vnfs"][0]["cpu"] = 999
        self.assertEqual(env.request.source, 0)
        self.assertEqual(env.request.vnfs[0].cpu, 100)

    def test_config_and_demo_integration(self):
        cfg = SimulationConfig.from_dict({"mdp": {"max_steps": 25}})
        self.assertEqual(cfg.constellation.satellites, 20)
        self.assertEqual(cfg.constellation.planes, 4)
        report = run_mdp_demo(cfg)
        self.assertTrue(report["trajectory"][-1]["info"]["accepted"])
        self.assertEqual(report["after_expiry"]["cpu_reserved"], 0)
        self.assertEqual(report["after_expiry"]["bandwidth_reserved_mhz"], 0)
        for kwargs in ({"max_steps": 0}, {"delay_scale_s": 0}):
            with self.assertRaises(ValueError):
                MDPConfig(**kwargs)
        with self.assertRaises(ValueError):
            RewardConfig(beta0=5, beta1=10)


if __name__ == "__main__":
    unittest.main()
