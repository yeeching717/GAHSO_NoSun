"""可行策略、正規化獎勵與持續資源訓練的回歸測試。"""
import json
import unittest
from pathlib import Path
from dataclasses import replace
import torch
from gahso.deployment import DeploymentPolicy
from gahso.features import GATFeatureBuilder
from gahso.mdp import SFCOrchestrationEnv
from gahso.resources import ResourceManager
from gahso.rollout import RolloutCollector, td_target
from gahso.simulation import SimulationConfig
from gahso.training import normalize_advantages, A2CTrainer
from gahso.evaluate import training_batch, advance_training_time, assess_convergence
from test_actor_critic import FixedConstellation, agent, request, route_forward


class StabilityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)
        cls.config = SimulationConfig.from_dict(json.loads(Path("config/default.json").read_text(encoding="utf-8")))

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def test_normalized_success_and_failure_returns(self):
        for capacity, delay, accepted in ((200, .1, True), (200, .005, False), (10, .1, False)):
            env = SFCOrchestrationEnv(FixedConstellation(capacity=capacity), self.config.mdp)
            model = agent()
            route_forward(model)
            roll = RolloutCollector(env, model).collect(request(delay=delay), deterministic=True)
            self.assertEqual(roll.accepted, accepted)
            self.assertGreater(roll.total_reward if accepted else -roll.total_reward, 0)
            self.assertTrue(all(abs(t.reward - sum(t.info["reward_components"].values())) < 1e-6 for t in roll.transitions))
            self.assertEqual(env.resources.metrics()["cpu_reserved"], 300 if accepted else 0)
            env.advance_time(10)
            self.assertEqual(env.resources.metrics()["cpu_reserved"], 0)
            self.assertEqual(env.resources.metrics()["bandwidth_reserved_mhz"], 0)

    def test_actor_zero_probability_for_visited_and_blocked_nodes(self):
        env = SFCOrchestrationEnv(FixedConstellation())
        env.reset(request=request())
        state, *_ = env.step(1)
        graph = GATFeatureBuilder().build(state)
        model = agent()
        probabilities = model.actor.distribution(graph).probs
        self.assertEqual(probabilities[0].item(), 0)
        self.assertEqual(probabilities[2].item(), 1)
        self.assertEqual(probabilities[~graph.feasible_action_mask].sum().item(), 0)

    def test_deployment_and_rollout_share_feasibility_and_no_action_finish(self):
        for capacity in (10, 200):
            topology = FixedConstellation(capacity=capacity)
            model = agent()
            route_forward(model)
            env = SFCOrchestrationEnv(topology)
            roll = RolloutCollector(env, model).collect(request(), deterministic=True)
            manager = ResourceManager(topology.snapshot(0))
            policy = DeploymentPolicy(model.actor)
            allocation = policy(manager, request())
            self.assertEqual(allocation is not None, roll.accepted)
            self.assertEqual(policy.last_report["info"]["reason"], roll.transitions[-1].info["reason"])
            self.assertEqual(manager.metrics()["bandwidth_reserved_mhz"], 0)
            if allocation:
                self.assertEqual(allocation.path, roll.transitions[-1].next_state["path"])

    def test_advantage_standardization_and_constant_fallback(self):
        normalized = normalize_advantages(torch.tensor([-2., 0., 4.], requires_grad=True))
        self.assertAlmostEqual(normalized.mean().item(), 0, places=6)
        self.assertAlmostEqual(normalized.std(unbiased=False).item(), 1, places=6)
        self.assertFalse(normalized.requires_grad)
        for values in ([0.], [2.], [-2., -2.]):
            result = normalize_advantages(torch.tensor(values))
            self.assertTrue(torch.isfinite(result).all())
            self.assertEqual(result.sign().tolist(), torch.tensor(values).sign().tolist())

    def test_scaled_targets_termination_and_external_truncation(self):
        self.assertEqual(td_target(1., torch.tensor(2.), False, .9).item(), torch.tensor(2.8).item())
        self.assertEqual(td_target(1., torch.tensor(float("nan")), True, .9).item(), 1.)
        env = SFCOrchestrationEnv(FixedConstellation(), self.config.mdp)
        trainer = A2CTrainer(env, agent(), self.config.training)
        report = trainer.train_request(request(), include_trajectory=True)
        self.assertAlmostEqual(report["reward_after_normalization"], report["total_reward"] / 5)
        for t in report["trajectory"]:
            expected = t["reward"] / 5 + (0 if t["terminated"] else trainer.agent.config.discount * t["next_value"])
            self.assertAlmostEqual(t["td_target"], expected, places=5)
        json.dumps(report, allow_nan=False)

    def test_persistent_batches_preserve_allocations_and_expire(self):
        first = training_batch(self.config, 3, 1, 1, "persistent", 0)
        second = training_batch(self.config, 3, 2, 2, "persistent", first[-1].arrival_s)
        self.assertGreater(second[0].arrival_s, first[-1].arrival_s)
        self.assertFalse({q.id for q in first} & {q.id for q in second})
        env = SFCOrchestrationEnv(FixedConstellation())
        env.reset(request=request())
        for node in (1, 2, 3):
            env.step(node)
        advance_training_time(env, 5, 2)
        self.assertEqual(env.resources.metrics()["cpu_reserved"], 300)
        advance_training_time(env, 11, 2)
        self.assertEqual(env.resources.metrics()["cpu_reserved"], 0)
        self.assertEqual(env.resources.metrics()["bandwidth_reserved_mhz"], 0)

    def test_convergence_is_invariant_to_loss_units(self):
        first = dict(mean_reward=-5., success_rate=.1, mean_critic_loss=.2,
                     mean_abs_td_error=.4, feasible_action_ratio=.7)
        middle = dict(first, mean_reward=-2., success_rate=.2, mean_critic_loss=.1, mean_abs_td_error=.2)
        last = dict(middle, mean_reward=-1.98, success_rate=.21, mean_critic_loss=.105, mean_abs_td_error=.205)
        history = [first] * 10 + [middle] * 5 + [last] * 5
        scaled = [dict(row, mean_critic_loss=row["mean_critic_loss"] * 100,
                       mean_abs_td_error=row["mean_abs_td_error"] * 10) for row in history]
        a, b = assess_convergence(history), assess_convergence(scaled)
        self.assertEqual(a["status"], "stable_window_not_proof")
        self.assertEqual(a["status"], b["status"])
        self.assertAlmostEqual(a["critic_loss_relative_change"], b["critic_loss_relative_change"])
        self.assertAlmostEqual(a["td_error_relative_change"], b["td_error_relative_change"])
