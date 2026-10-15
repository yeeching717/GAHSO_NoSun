"""Actor/Critic、目標同步與逐跳獎勵的整合測試。"""
import unittest
from dataclasses import replace
import torch
from gahso.actor_critic import ActorCriticAgent, NoAvailableActionsError
from gahso.agent_config import AgentConfig
from gahso.gat_config import GATConfig
from gahso.features import GATFeatureBuilder
from gahso.rollout import RolloutCollector, td_target
from gahso.mdp import SFCOrchestrationEnv, MDPConfig
from gahso.network import ConstellationConfig, Satellite, Link, NetworkSnapshot, edge_key
from gahso.services import SFCRequest, VNF


def setUpModule():
    global previous_threads
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)


def tearDownModule():
    torch.set_num_threads(previous_threads)


class FixedConstellation:
    config = ConstellationConfig()

    def __init__(self, edges=((0, 1), (1, 2), (2, 3)), capacity=200, cpu=500):
        self.edges, self.capacity, self.cpu = edges, capacity, cpu

    def snapshot(self, time_s):
        nodes = {n: Satellite(n, 0, n, (8000, n * 10, 0), self.cpu) for n in range(4)}
        links = {edge_key(a, b): Link(*edge_key(a, b), 2990, .01, self.capacity, "test")
                 for a, b in self.edges}
        return NetworkSnapshot(time_s, nodes, links)


def request(identifier="q", arrival=0, delay=.1, cpu=100):
    return SFCRequest(identifier, 0, 3, tuple(VNF(str(i), cpu) for i in range(3)),
                      50, delay, arrival, 10)


def agent():
    return ActorCriticAgent(4, gat_config=GATConfig(hidden_dim=8, num_layers=1),
                           config=AgentConfig(hidden_dim1=16, hidden_dim2=8, target_sync_interval=2))


def route_forward(model):
    # 固定偏好較大的 node index，讓鏈狀拓樸沿 0→1→2→3 行走。
    with torch.no_grad():
        for p in model.actor.parameters():
            p.zero_()
        model.actor.head[-1].bias.copy_(torch.arange(4.))


class NetworkTests(unittest.TestCase):
    def setUp(self):
        self.env = SFCOrchestrationEnv(FixedConstellation())
        state, _ = self.env.reset(request=request())
        self.graph = GATFeatureBuilder().build(state)
        self.agent = agent()

    def test_mask_and_single_neighbor_distribution(self):
        sample = self.agent.act(self.graph)
        self.assertEqual(sample.action, 1)
        torch.testing.assert_close(sample.probabilities, torch.tensor([0., 1., 0., 0.]))
        self.assertAlmostEqual(sample.log_prob.item(), 0)
        self.assertAlmostEqual(sample.entropy.item(), 0)
        self.assertEqual(self.agent.critic(self.graph).shape, torch.Size([]))

    def test_all_masked_actor_rejects_but_critic_works(self):
        graph = replace(self.graph, feasible_action_mask=torch.zeros(4, dtype=torch.bool))
        with self.assertRaises(NoAvailableActionsError):
            self.agent.act(graph)
        self.assertTrue(torch.isfinite(self.agent.critic(graph)))

    def test_independent_parameters_and_frozen_target(self):
        actor = list(self.agent.actor.parameters())
        critic = list(self.agent.critic.parameters())
        target = list(self.agent.target_critic.parameters())
        self.assertTrue(set(p.data_ptr() for p in actor).isdisjoint(p.data_ptr() for p in critic))
        for p, q in zip(critic, target):
            torch.testing.assert_close(p, q)
            self.assertNotEqual(p.data_ptr(), q.data_ptr())
            self.assertFalse(q.requires_grad)
        self.agent.train()
        self.assertFalse(self.agent.target_critic.training)
        self.assertFalse(self.agent.target_critic.value_network.training)
        self.assertFalse(self.agent.target_critic(self.graph).requires_grad)

    def test_target_only_changes_at_sync_interval(self):
        target = next(self.agent.target_critic.parameters())
        before = target.clone()
        with torch.no_grad():
            next(self.agent.critic.parameters()).add_(1)
        self.assertFalse(self.agent.notify_critic_updated())
        torch.testing.assert_close(target, before)
        self.assertTrue(self.agent.notify_critic_updated())
        torch.testing.assert_close(target, next(self.agent.critic.parameters()))
        restored = agent()
        restored.load_state_dict(self.agent.state_dict())
        self.assertEqual(restored.critic_updates.item(), 2)

    def test_explicit_sync_and_config_mismatch(self):
        with torch.no_grad():
            self.agent.critic.head[-1].bias.add_(3)
        self.agent.sync_target()
        self.assertEqual(self.agent.critic_updates.item(), 0)
        torch.testing.assert_close(self.agent.critic(self.graph), self.agent.target_critic(self.graph))
        other = ActorCriticAgent(5, gat_config=GATConfig(hidden_dim=8, num_layers=1))
        with self.assertRaises(ValueError):
            self.agent.target_critic.sync_from(other.critic)

    def test_twenty_satellite_output(self):
        env = SFCOrchestrationEnv()
        state, _ = env.reset(request=replace(request(), destination=2))
        graph = GATFeatureBuilder().build(state)
        model = ActorCriticAgent()
        probs = model.actor.distribution(graph).probs
        self.assertEqual(probs.shape, (25,))
        self.assertTrue(torch.equal(probs[~graph.action_mask], torch.zeros_like(probs[~graph.action_mask])))
        self.assertAlmostEqual(probs.sum().item(), 1, places=6)

    def test_invalid_config(self):
        for kwargs in ({"discount": float("nan")}, {"discount": 1.1}, {"hidden_dim1": 0},
                       {"target_sync_interval": 0}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                AgentConfig(**kwargs)


class TDTests(unittest.TestCase):
    def test_nonterminal_target_detaches(self):
        value = torch.tensor(4., requires_grad=True)
        target = td_target(2, value, False, .5)
        self.assertEqual(target.item(), 4)
        self.assertFalse(target.requires_grad)

    def test_terminal_does_not_bootstrap_even_nan(self):
        self.assertEqual(td_target(-10, torch.tensor(float("nan")), True, .99).item(), -10)

    def test_nonterminal_rejects_nonfinite(self):
        with self.assertRaises(ValueError):
            td_target(1, torch.tensor(float("nan")), False, .99)


class RolloutTests(unittest.TestCase):
    def make(self, **kwargs):
        env = SFCOrchestrationEnv(FixedConstellation(**kwargs), MDPConfig(delay_feasibility_mask=False))
        model = agent()
        route_forward(model)
        return env, model, RolloutCollector(env, model)

    def test_success_trajectory_rewards_and_resource_expiry(self):
        env, model, collector = self.make()
        result = collector.collect(request(), deterministic=True)
        self.assertTrue(result.accepted)
        self.assertEqual([t.action for t in result.transitions], [1, 2, 3])
        self.assertEqual(result.transitions[-1].next_state["path"], (0, 1, 2, 3))
        for step in result.transitions:
            self.assertAlmostEqual(step.reward, sum(step.info["reward_components"].values()))
            self.assertFalse(step.value.requires_grad)
            self.assertEqual(step.action_node_id, step.action)
        last = result.transitions[-1]
        self.assertTrue(last.terminated)
        self.assertEqual(last.next_value.item(), 0)
        self.assertAlmostEqual(last.target.item(), last.reward, places=4)
        self.assertEqual(env.resources.metrics()["cpu_reserved"], 300)
        self.assertEqual(model.critic_updates.item(), 0)
        env.advance_time(10)
        self.assertEqual(env.resources.metrics()["cpu_reserved"], 0)

    def test_gradient_flow_and_training_mode_restoration(self):
        env, model, collector = self.make()
        model.train()
        result = collector.collect(request(), deterministic=True, track_grad=True)
        loss = sum((t.value - t.target).square() -
                   t.log_prob * t.advantage - .01 * t.entropy for t in result.transitions)
        loss.backward()
        self.assertTrue(model.actor.training)
        self.assertTrue(model.critic.training)
        self.assertFalse(model.target_critic.training)
        self.assertTrue(any(p.grad is not None for p in model.actor.parameters()))
        self.assertTrue(any(p.grad is not None for p in model.critic.parameters()))
        self.assertTrue(all(p.grad is None for p in model.target_critic.parameters()))
        self.assertTrue(all(not t.advantage.requires_grad for t in result.transitions))
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))

    def test_background_allocations_survive_next_request(self):
        env, _, collector = self.make()
        collector.collect(request(), deterministic=True)
        second = collector.collect(request("q2", arrival=1), deterministic=True)
        self.assertTrue(second.accepted)
        self.assertEqual(env.resources.metrics()["cpu_reserved"], 600)
        env.advance_time(10)
        self.assertEqual(env.resources.metrics()["cpu_reserved"], 300)
        env.advance_time(11)
        self.assertEqual(env.resources.metrics()["cpu_reserved"], 0)

    def test_delay_failure_does_not_reserve_resources(self):
        env, _, collector = self.make()
        result = collector.collect(request(delay=.005), deterministic=True)
        self.assertFalse(result.accepted)
        self.assertEqual(result.transitions[-1].info["outcome"], "delay_violation")
        self.assertEqual(env.resources.metrics()["cpu_reserved"], 0)

    def test_embedding_failure_does_not_reserve_resources(self):
        env, _, collector = self.make(cpu=50)
        result = collector.collect(request(), deterministic=True)
        self.assertFalse(result.accepted)
        self.assertEqual(result.transitions[-1].info["reason"], "vnf_embedding_failed")
        self.assertEqual(env.resources.metrics()["cpu_reserved"], 0)

    def test_step_limit_is_terminal(self):
        env = SFCOrchestrationEnv(FixedConstellation(), MDPConfig(max_steps=1))
        model = agent()
        result = RolloutCollector(env, model).collect(request())
        step = result.transitions[0]
        self.assertTrue(step.terminated)
        self.assertFalse(step.truncated)
        self.assertEqual(step.info["reason"], "max_steps")
        self.assertAlmostEqual(step.target.item(), step.reward, places=4)
        self.assertIn("instant_reward_components", step.info)

    def test_isolated_start_has_no_fabricated_action(self):
        env, _, collector = self.make(edges=())
        result = collector.collect(request())
        step = result.transitions[0]
        self.assertIsNone(step.action)
        self.assertIsNone(step.log_prob)
        self.assertTrue(step.terminated)
        self.assertEqual(result.to_dict()["policy_steps"], 0)
        self.assertEqual(env.steps, 0)
        self.assertTrue(torch.isfinite(step.target))

    def test_bandwidth_infeasible_neighbor_is_masked(self):
        _, _, collector = self.make(capacity=10)
        result = collector.collect(request())
        step = result.transitions[0]
        self.assertIsNone(step.action)
        self.assertIsNone(step.probabilities)
        self.assertEqual(step.info["reason"], "dead_end")
        self.assertFalse(result.accepted)

    def test_invalid_revisit_penalty_then_step_limit(self):
        env = SFCOrchestrationEnv(FixedConstellation(), MDPConfig(max_steps=3))
        model = agent()
        route_forward(model)
        with torch.no_grad():
            model.actor.head[-1].bias[0] = 10
        result = RolloutCollector(env, model).collect(request(), deterministic=True)
        self.assertEqual([t.action for t in result.transitions], [1, 2, 3])
        self.assertEqual(result.transitions[1].probabilities[0].item(), 0)
        self.assertTrue(result.accepted)

    def test_seeded_sampling_and_no_inference_update(self):
        env1, model1, collector1 = self.make()
        env2, model2, collector2 = self.make()
        model2.load_state_dict(model1.state_dict())
        before = {k: v.clone() for k, v in model1.state_dict().items()}
        rng = torch.random.get_rng_state()
        first = collector1.collect(request())
        second = collector2.collect(request())
        self.assertEqual([t.action for t in first.transitions], [t.action for t in second.transitions])
        torch.testing.assert_close(torch.random.get_rng_state(), rng)
        for key, value in model1.state_dict().items():
            torch.testing.assert_close(value, before[key])

    def test_no_action_finish_rejects_when_neighbors_exist(self):
        env, _, _ = self.make()
        env.reset(request=request())
        with self.assertRaises(ValueError):
            env.finish_without_action()


if __name__ == "__main__":
    unittest.main()
