"""部署凍結／checkpoint，以及遷移最优性、原子帳本、動態指標的可驗算案例。"""
import io
import json
import tempfile
import unittest
from dataclasses import replace
from itertools import permutations, combinations_with_replacement
from pathlib import Path
import torch
from gahso.deployment import DeploymentPolicy, FrozenTopology, save_policy, load_policy
from gahso.dynamic import FaultTopology, run_dynamic, validate_active
from gahso.evaluate import assess_convergence
from gahso.mdp import MDPConfig
from gahso.migration import MigrationConfig, MigrationPlanner
from gahso.network import NetworkSnapshot
from gahso.resources import Allocation, ResourceManager, ServiceStatus, jain_fairness
from gahso.simulation import SimulationConfig
from test_actor_critic import FixedConstellation, agent, request, route_forward


def setUpModule():
    global previous_threads
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)


def tearDownModule():
    torch.set_num_threads(previous_threads)


def diamond(time=0, edges=((0, 1), (1, 3), (0, 2), (2, 3)), capacity=200):
    return FixedConstellation(edges, capacity=capacity).snapshot(time)


class DeploymentTests(unittest.TestCase):
    def test_frozen_topology_keeps_links_but_advances_time(self):
        original = FixedConstellation().snapshot(0)
        later = FrozenTopology(original).snapshot(17)
        self.assertEqual(later.time_s, 17)
        self.assertEqual(later.links, original.links)

    def test_inference_is_read_only_and_matches_known_route(self):
        model = agent()
        route_forward(model)
        model.train()
        policy = DeploymentPolicy(model.actor)
        manager = ResourceManager(FixedConstellation().snapshot(1))
        q = request(arrival=1)
        before = manager.metrics()
        allocation = policy(manager, q)
        self.assertEqual(allocation, Allocation((0, 1, 2, 3), (0, 1, 3)))
        self.assertEqual(manager.metrics(), before)
        self.assertEqual(manager.events, [])
        self.assertTrue(model.actor.training)
        self.assertTrue(all(not p.requires_grad and p.grad is None for p in policy.actor.parameters()))
        self.assertTrue(manager.admit(q, allocation).accepted)
        validate_active(manager)

    def test_checkpoint_roundtrip_and_independence(self):
        model = agent().double()
        route_forward(model)
        policy = DeploymentPolicy(model.actor)
        rng = torch.random.get_rng_state()
        with tempfile.TemporaryDirectory() as directory:
            filename = Path(directory) / "policy.pt"
            save_policy(filename, model)
            loaded = load_policy(filename)
        torch.testing.assert_close(torch.random.get_rng_state(), rng)
        self.assertEqual(next(loaded.actor.parameters()).dtype, torch.float64)
        for p, q in zip(policy.actor.parameters(), loaded.actor.parameters()):
            torch.testing.assert_close(p, q)
        with torch.no_grad():
            next(model.actor.parameters()).add_(1)
        self.assertFalse(torch.equal(next(model.actor.parameters()), next(policy.actor.parameters())))
        manager = ResourceManager(FixedConstellation().snapshot(0))
        self.assertEqual(loaded(manager, request()), policy(manager, request()))

    def test_checkpoint_preserves_endpoint_mode(self):
        model = agent()
        with tempfile.TemporaryDirectory() as directory:
            filename = Path(directory) / "neighbor.pt"
            save_policy(filename, model, MDPConfig(endpoint_placement_mode="neighbor_allowed"))
            loaded = load_policy(filename)
        self.assertEqual(loaded.mdp_config.endpoint_placement_mode, "neighbor_allowed")

    def test_failed_inference_does_not_reject_on_real_ledger(self):
        policy = DeploymentPolicy(agent().actor)
        manager = ResourceManager(FixedConstellation(edges=()).snapshot(0))
        self.assertIsNone(policy(manager, request()))
        self.assertEqual(manager.records, {})
        self.assertEqual(manager.metrics()["cpu_reserved"], 0)


class MigrationTests(unittest.TestCase):
    def initial(self):
        manager = ResourceManager(diamond())
        self.assertTrue(manager.admit(request(), Allocation((0, 1, 3), (0, 1, 3))).accepted)
        return manager

    def test_one_vnf_move_and_exact_release(self):
        manager = self.initial()
        manager.update(diamond(1, edges=((0, 2), (2, 3))))
        result = MigrationPlanner().repair(manager, "q")
        self.assertTrue(result["repaired"])
        self.assertTrue(result["search_exhausted"])
        self.assertEqual(result["vnf_migrations"], 1)
        self.assertTrue(result["rerouted"])
        self.assertEqual(manager.cpu_available(1), 500)
        self.assertEqual(manager.metrics()["cpu_reserved"], 300)
        validate_active(manager)
        manager.update(diamond(10))
        self.assertEqual(manager.records["q"].status, ServiceStatus.EXPIRED)
        self.assertEqual(manager.metrics()["cpu_reserved"], 0)
        self.assertEqual(manager.metrics()["bandwidth_reserved_mhz"], 0)
        self.assertFalse(manager.release("q"))

    def test_reroute_without_vnf_migration(self):
        manager = self.initial()
        manager.update(diamond(1, edges=((0, 1), (1, 2), (2, 3))))
        result = MigrationPlanner().repair(manager, "q")
        self.assertTrue(result["repaired"])
        self.assertEqual(result["vnf_migrations"], 0)
        self.assertTrue(result["rerouted"])
        self.assertEqual(manager.records["q"].allocation.vnf_nodes, (0, 1, 3))

    def test_infeasible_migration_is_atomic(self):
        manager = self.initial()
        manager.update(diamond(1, edges=()))
        before = manager.metrics()
        old = manager.records["q"].allocation
        result = MigrationPlanner().repair(manager, "q")
        self.assertFalse(result["repaired"])
        self.assertEqual(manager.metrics(), before)
        self.assertEqual(manager.records["q"].allocation, old)
        manager.assert_invariants()

    def test_bandwidth_and_delay_prevent_repair(self):
        for snapshot in (diamond(1, edges=((0, 2), (2, 3)), capacity=10),
                         NetworkSnapshot(1, diamond().satellites,
                                         {key: replace(link, delay_s=.2) for key, link in
                                          diamond(edges=((0, 2), (2, 3))).links.items()})):
            manager = self.initial()
            manager.update(snapshot)
            self.assertFalse(MigrationPlanner().repair(manager, "q")["repaired"])
            manager.assert_invariants()

    def test_own_resource_credit_and_background_are_separate(self):
        manager = ResourceManager(diamond(capacity=50))
        manager.admit(request(), Allocation((0, 1, 3), (0, 1, 3)))
        manager.update(diamond(1, edges=((0, 1), (1, 2), (2, 3)), capacity=50))
        self.assertEqual(manager.bandwidth_available(0, 1), 0)
        view = manager.planning_copy("q")
        self.assertEqual(view.bandwidth_available(0, 1), 50)
        self.assertEqual(manager.bandwidth_available(0, 1), 0)
        self.assertTrue(MigrationPlanner().repair(manager, "q")["repaired"])
        self.assertEqual(manager.bandwidth_available(0, 1), 0)
        validate_active(manager)

    def test_search_limit_reported(self):
        manager = self.initial()
        manager.update(diamond(1, edges=((0, 2), (2, 3))))
        result = MigrationPlanner(MigrationConfig(max_expansions=1)).repair(manager, "q")
        self.assertFalse(result["search_exhausted"])
        self.assertEqual(result["reason"], "search_limit")

    def test_matches_independent_brute_force_small_graph(self):
        manager = self.initial()
        manager.update(diamond(1, edges=((0, 2), (2, 1), (1, 3), (2, 3))))
        old = manager.records["q"].allocation
        q = request()
        view = manager.planning_copy("q")
        costs = []
        for count in range(3):
            for middle in permutations((1, 2), count):
                path = (0,) + middle + (3,)
                for node in path:
                    allocation = Allocation(path, (0, node, 3))
                    if view.check_allocation(q, allocation).feasible:
                        fairness = jain_fairness(view.projected_loads(q, allocation).values())
                        costs.append(sum(a != b for a, b in zip(old.vnf_nodes, allocation.vnf_nodes)) - 1 - fairness)
        result = MigrationPlanner().repair(manager, "q")
        self.assertTrue(result["search_exhausted"])
        self.assertAlmostEqual(result["objective"], min(costs))

    def test_invalid_order_replace_rejected_without_changes(self):
        manager = self.initial()
        old = manager.metrics()
        q4 = replace(request("q4"), vnfs=request().vnfs + (request().vnfs[-1],))
        view = manager.planning_copy()
        self.assertFalse(view.check_allocation(q4, Allocation((0, 1, 3), (0, 3, 1, 3))).feasible)
        self.assertFalse(manager.replace_allocation("q", Allocation((0, 1, 3), (0, 3, 1))).accepted)
        self.assertEqual(manager.metrics(), old)


class ChangingTopology:
    def snapshot(self, time_s):
        edges = ((0, 1), (1, 3)) if time_s < 5 else ((0, 2), (2, 3))
        return diamond(time_s, edges=edges)


class DynamicTests(unittest.TestCase):
    def test_time_availability_and_migration_metrics(self):
        cfg = SimulationConfig(horizon_s=20, topology_step_s=5)
        q = replace(request(), duration_s=20)
        policy = lambda manager, q: Allocation((0, 1, 3), (0, 1, 3))
        no = run_dynamic(cfg, [q], policy, topology=ChangingTopology(), repair=False)
        yes = run_dynamic(cfg, [q], policy, topology=ChangingTopology(), repair=True)
        self.assertAlmostEqual(no["service_time_availability"], .25)
        self.assertAlmostEqual(yes["service_time_availability"], 1)
        self.assertEqual(yes["vnf_migrations"], 1)
        self.assertEqual(yes["reroutes"], 1)
        self.assertEqual(no["uninterrupted_after_repair_rate"], 0)
        self.assertEqual(yes["uninterrupted_after_repair_rate"], 1)
        self.assertEqual(yes["final"]["cpu_reserved"], 0)
        self.assertEqual(yes["event_continuity_after_repair"], 1)
        json.dumps(yes, allow_nan=False)

    def test_no_admitted_service_is_not_fake_hundred_percent(self):
        result = run_dynamic(SimulationConfig(horizon_s=20), [request()],
                             lambda manager, q: None, topology=ChangingTopology())
        self.assertIsNone(result["service_time_availability"])
        self.assertIsNone(result["event_continuity_after_repair"])
        self.assertIsNone(result["uninterrupted_after_repair_rate"])

    def test_fault_seed_and_extremes(self):
        constellation = FixedConstellation()
        a = FaultTopology(constellation, probability=.5, seed=17)
        b = FaultTopology(constellation, probability=.5, seed=17)
        for time in (0, 1, 10, 15, 20):
            self.assertEqual(dict(a.snapshot(time).links), dict(b.snapshot(time).links))
        self.assertEqual(len(FaultTopology(constellation, probability=1).snapshot(0).links), 0)
        self.assertEqual(len(FaultTopology(constellation, probability=0).snapshot(0).links), 3)
        with self.assertRaises(ValueError):
            FaultTopology(constellation, probability=-1)

    def test_short_experiment_is_not_declared_converged(self):
        self.assertEqual(assess_convergence([])["status"], "insufficient_epochs")
        history = [{"mean_reward": -100, "mean_critic_loss": 100, "success_rate": .1}] * 20
        self.assertEqual(assess_convergence(history)["status"], "not_converged")


if __name__ == "__main__":
    unittest.main()
