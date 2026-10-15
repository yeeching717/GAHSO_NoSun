"""請求層級更新的時序、數學方向、資源與跨請求梯度隔離。"""
import json
import unittest
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch
import torch
from gahso.training import A2CTrainer, request_losses, batch_request_losses
from gahso.training_config import TrainingConfig
from gahso.rollout import RolloutCollector
from gahso.mdp import SFCOrchestrationEnv, MDPConfig
from gahso.simulation import SimulationConfig
from helpers import check_training_updates
from test_actor_critic import FixedConstellation, agent, request, route_forward


def setUpModule():
    global previous_threads
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)


def tearDownModule():
    torch.set_num_threads(previous_threads)


def make(**kwargs):
    env = SFCOrchestrationEnv(FixedConstellation(**kwargs), MDPConfig(delay_feasibility_mask=False))
    model = agent()
    route_forward(model)
    trainer = A2CTrainer(env, model)
    return env, model, trainer


def greedy(model):
    original = model.act
    return patch.object(model, "act", side_effect=lambda graph, **kw: original(graph, deterministic=True))


class TrainingTests(unittest.TestCase):
    def test_batch_advantage_update_is_finite(self):
        env, model, trainer = make()
        rollouts = []
        with greedy(model):
            rollouts.append(trainer.collector.collect(request(), track_grad=True))
            rollouts.append(trainer.collector.collect(request("q2", arrival=1), track_grad=True))
        losses = batch_request_losses(rollouts, trainer.config.entropy_coefficient)
        self.assertTrue(all(torch.isfinite(x) for x in (losses.actor, losses.critic, losses.entropy)))
        reports = trainer.train_rollouts(rollouts)
        self.assertEqual(len(reports), 2)
        self.assertEqual(trainer.requests_updated, 2)
        self.assertEqual(model.critic_updates.item(), 1)
        self.assertTrue(all(r["initially_feasible"] for r in reports))

    def test_training_batch_config_validation(self):
        self.assertEqual(TrainingConfig().update_requests_per_batch, 8)
        with self.assertRaises(ValueError):
            TrainingConfig(update_requests_per_batch=0)

    def test_updates_only_after_embedding_and_terminal_reward(self):
        env, model, trainer = make()
        before = {key: value.clone() for key, value in model.state_dict().items()}
        original_step = env.step
        def observe_step(action):
            for key, value in model.state_dict().items():
                torch.testing.assert_close(value, before[key])
            self.assertEqual(env.resources.metrics()["cpu_reserved"], 0)
            result = original_step(action)
            if result[2]:
                self.assertEqual(result[-1]["embedding"]["placements"], (0, 1, 3))
                self.assertEqual(env.resources.metrics()["cpu_reserved"], 300)
                self.assertAlmostEqual(result[1], sum(result[-1]["reward_components"].values()))
            return result
        original_actor_step = trainer.actor_optimizer.step
        def observe_update(*args, **kwargs):
            self.assertEqual(env.observation()["path"], (0, 1, 2, 3))
            self.assertFalse(any(env.observation()["action_mask"]))
            self.assertEqual(env.resources.metrics()["cpu_reserved"], 300)
            return original_actor_step(*args, **kwargs)
        with greedy(model), patch.object(env, "step", side_effect=observe_step), \
                patch.object(trainer.actor_optimizer, "step", side_effect=observe_update) as actor_step, \
                patch.object(trainer.critic_optimizer, "step", wraps=trainer.critic_optimizer.step) as critic_step:
            report = trainer.train_request(request(), include_trajectory=True)
        self.assertEqual(actor_step.call_count, 1)
        self.assertEqual(critic_step.call_count, 1)
        self.assertEqual(report["steps"], 3)
        self.assertTrue(report["accepted"])
        # 此鏈狀拓樸每步只有一個可行動作，log pi=0，不應製造策略梯度。
        self.assertFalse(any(not torch.equal(value, before["actor." + key])
                            for key, value in model.actor.state_dict().items()))
        self.assertTrue(any(not torch.equal(value, before["critic." + key])
                            for key, value in model.critic.state_dict().items()))
        self.assertTrue(all(p.grad is None for p in model.parameters()))
        self.assertAlmostEqual(report["trajectory"][-1]["td_target"],
                               report["terminal_reward"] / trainer.config.reward_scale, places=4)
        json.dumps(report, allow_nan=False)

    def test_two_requests_sync_once_and_keep_resources(self):
        env, model, trainer = make()
        with greedy(model):
            first = trainer.train_request(request())
            target = {k: p.clone() for k, p in model.target_critic.state_dict().items()}
            second = trainer.train_request(request("q2", arrival=1))
        self.assertFalse(first["target_synced"])
        self.assertTrue(second["target_synced"])
        self.assertEqual(second["requests_updated"], 2)
        self.assertEqual(second["actor_updates"], 2)
        self.assertEqual(second["critic_updates"], 2)
        self.assertEqual(env.resources.metrics()["cpu_reserved"], 600)
        self.assertTrue(any(not torch.equal(v, target[k]) for k, v in model.target_critic.state_dict().items()))
        for p, q in zip(model.critic.parameters(), model.target_critic.parameters()):
            torch.testing.assert_close(p, q)
            self.assertFalse(q.requires_grad)
        env.advance_time(11)
        self.assertEqual(env.resources.metrics()["cpu_reserved"], 0)

    def test_no_cross_request_gradient_accumulation(self):
        # 與手動「每次 zero_grad → sum loss → backward → step」的兩次更新比較。
        env, model, trainer = make()
        ref_env, ref_model, ref_trainer = make()
        ref_model.load_state_dict(model.state_dict())
        with greedy(model), greedy(ref_model):
            for q in (request(), request("q2", arrival=1)):
                trainer.train_request(q)
                ref_trainer.actor_optimizer.zero_grad(set_to_none=True)
                ref_trainer.critic_optimizer.zero_grad(set_to_none=True)
                roll = ref_trainer.collector.collect(q, track_grad=True)
                loss = request_losses(roll, ref_trainer.config.entropy_coefficient)
                (loss.actor + loss.critic).backward()
                for network in (ref_model.actor, ref_model.critic):
                    torch.nn.utils.clip_grad_norm_(network.parameters(), ref_trainer.config.max_grad_norm)
                ref_trainer.actor_optimizer.step()
                ref_trainer.critic_optimizer.step()
                ref_model.notify_critic_updated()
        for key, value in model.state_dict().items():
            torch.testing.assert_close(value, ref_model.state_dict()[key])

    def test_failure_requests_still_update_once(self):
        for kwargs, q, expected in (({"cpu": 50}, request(), "vnf_embedding_failed"),
                                    ({}, request(delay=.005), "delay_exceeded"),
                                    ({"capacity": 10}, request(), "dead_end")):
            with self.subTest(reason=expected):
                env, model, trainer = make(**kwargs)
                with greedy(model), patch.object(trainer.critic_optimizer, "step",
                                                wraps=trainer.critic_optimizer.step) as step:
                    report = trainer.train_request(q)
                self.assertEqual(step.call_count, 1)
                self.assertEqual(report["critic_updates"], 1)
                self.assertFalse(report["accepted"])
                self.assertEqual(report["terminal_info"]["reason"], expected)
                self.assertIsNotNone(report["terminal_info"]["embedding"])
                self.assertEqual(env.resources.metrics()["cpu_reserved"], 0)

    def test_max_steps_failure_updates(self):
        env = SFCOrchestrationEnv(FixedConstellation(), MDPConfig(max_steps=1))
        trainer = A2CTrainer(env, agent())
        report = trainer.train_request(request())
        self.assertEqual(report["terminal_info"]["reason"], "max_steps")
        self.assertEqual(report["critic_updates"], 1)

    def test_isolated_request_updates_only_critic(self):
        env, model, trainer = make(edges=())
        before = {k: p.clone() for k, p in model.actor.state_dict().items()}
        with patch.object(trainer.actor_optimizer, "step", wraps=trainer.actor_optimizer.step) as step:
            report = trainer.train_request(request())
        self.assertEqual(step.call_count, 0)
        self.assertFalse(report["actor_updated"])
        self.assertEqual(report["critic_updates"], 1)
        self.assertEqual(report["actor_loss"], 0)
        for k, p in model.actor.state_dict().items():
            torch.testing.assert_close(p, before[k])

    def test_duplicate_request_does_not_update_again(self):
        _, _, trainer = make()
        trainer.train_request(request())
        with self.assertRaises(ValueError):
            trainer.train_request(request())
        self.assertEqual(trainer.requests_updated, 1)
        self.assertEqual(trainer.agent.critic_updates.item(), 1)

    def test_loss_mean_and_positive_advantage_direction(self):
        logit = torch.tensor(0., requires_grad=True)
        value = torch.tensor(0., requires_grad=True)
        target = torch.tensor(2., requires_grad=True)
        dist = torch.distributions.Categorical(logits=torch.stack((logit, logit.new_zeros(()))))
        step = SimpleNamespace(value=value, target=target, advantage=target-value,
                               action=0, log_prob=dist.log_prob(torch.tensor(0)),
                               entropy=dist.entropy(), terminated=False, truncated=False)
        terminal = SimpleNamespace(**{**vars(step), "terminated": True})
        losses = request_losses(SimpleNamespace(transitions=(step, terminal)))
        self.assertAlmostEqual(losses.critic.item(), 2)
        self.assertAlmostEqual(losses.actor.item(), torch.log(torch.tensor(2.)).item())
        (losses.actor + losses.critic).backward()
        self.assertLess(logit.grad.item(), 0)  # gradient descent 提高被選動作的 logit
        self.assertAlmostEqual(value.grad.item(), -2)
        self.assertIsNone(target.grad)

    def test_inference_rollout_cannot_train(self):
        env, model, _ = make()
        rollout = RolloutCollector(env, model).collect(request())
        with self.assertRaises(ValueError):
            request_losses(rollout)

    def test_nonfinite_gradient_prevents_both_updates(self):
        _, model, trainer = make()
        before = {k: p.clone() for k, p in model.state_dict().items()}
        hook = model.critic.head[-1].bias.register_hook(lambda grad: grad * float("nan"))
        try:
            with self.assertRaises(FloatingPointError):
                trainer.train_request(request())
        finally:
            hook.remove()
        for k, p in model.state_dict().items():
            torch.testing.assert_close(p, before[k])
        self.assertEqual(trainer.requests_updated, 0)
        self.assertTrue(all(p.grad is None for p in model.parameters()))

    def test_config_and_real_twenty_five_node_updates(self):
        cfg = SimulationConfig.from_dict({"training": {"max_grad_norm": 1.0}})
        report = check_training_updates(cfg)
        self.assertEqual(report["requests_updated"], 3)
        self.assertEqual(report["critic_updates"], 3)
        self.assertEqual(report["actor_updates"], 3)
        json.dumps(report, allow_nan=False)
        for kwargs in ({"actor_lr": 0}, {"critic_lr": float("nan")},
                       {"entropy_coefficient": -1}, {"max_grad_norm": 0}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                TrainingConfig(**kwargs)


if __name__ == "__main__":
    unittest.main()
