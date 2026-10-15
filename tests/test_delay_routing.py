"""Delay lower bounds, failure attribution, 25-node topology and checkpoint schema."""
import tempfile
import unittest
from dataclasses import replace
from itertools import permutations
from pathlib import Path
import torch
from gahso.network import LEOConstellation, ConstellationConfig, NetworkSnapshot, minimum_distance_matching
from gahso.resources import ResourceManager, Allocation
from gahso.routing import remaining_delays, arrival_diagnostics
from gahso.services import RequestGenerator, RequestConfig, SFCRequest, VNF
from gahso.mdp import SFCOrchestrationEnv, MDPConfig
from gahso.rollout import RolloutCollector
from gahso.deployment import DeploymentPolicy, save_policy, load_policy
from gahso.actor_critic import ActorCriticAgent
from gahso.topology_audit import audit_topology
from gahso.dynamic import run_dynamic
from gahso.simulation import SimulationConfig
from gahso.embedding import embed_vnfs
from test_actor_critic import FixedConstellation, request, agent, route_forward
from test_deployment import ChangingTopology


class Diamond(FixedConstellation):
    def __init__(self):
        super().__init__(edges=((0, 1), (1, 3), (0, 2), (2, 3)))

    def snapshot(self, time_s):
        snap = super().snapshot(time_s)
        return NetworkSnapshot(time_s, snap.satellites,
            {edge: replace(link, delay_s=.07 if 1 in edge else .01) for edge, link in snap.links.items()})


class DelayTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def test_actor_avoids_long_detour_and_deployment_matches(self):
        model = agent()
        route_forward(model)
        with torch.no_grad():
            model.actor.head[-1].bias[1] = 20
        for enabled in (True, False):
            cfg = MDPConfig(delay_feasibility_mask=enabled)
            env = SFCOrchestrationEnv(Diamond(), cfg)
            roll = RolloutCollector(env, model).collect(request(), deterministic=True)
            self.assertEqual(roll.accepted, enabled)
            self.assertEqual(roll.transitions[0].action, 2 if enabled else 1)
            if not enabled:
                self.assertEqual(roll.transitions[-1].info["failure_category"], "policy_delay_failure")
            policy = DeploymentPolicy(model.actor, cfg)
            allocation = policy(ResourceManager(Diamond().snapshot(0)), request())
            self.assertEqual(allocation is not None, enabled)
            self.assertEqual(policy.last_report["info"]["failure_category"], roll.transitions[-1].info["failure_category"])

    def test_bound_excludes_visited_and_bandwidth_blocked_edges(self):
        manager = ResourceManager(FixedConstellation(edges=((0,1),(1,2),(2,0),(0,3))).snapshot(0))
        self.assertIn(2, remaining_delays(manager, 3, 50))
        self.assertNotIn(2, remaining_delays(manager, 3, 50, excluded=(0,1)))
        blocked = ResourceManager(FixedConstellation(capacity=10).snapshot(0))
        self.assertNotIn(0, remaining_delays(blocked, 3, 50))

    def test_accumulated_delay_is_part_of_next_action_bound(self):
        class BudgetTopology(FixedConstellation):
            def snapshot(self, time_s):
                snap = super().snapshot(time_s)
                return NetworkSnapshot(time_s, snap.satellites,
                    {edge: replace(link, delay_s=.04) for edge, link in snap.links.items()})
        env = SFCOrchestrationEnv(BudgetTopology(edges=((0,1),(1,2),(2,3),(1,3))))
        state, _ = env.reset(request=request(delay=.1))
        self.assertTrue(state["feasible_action_mask"][1])
        env.step(1)
        # .04 已用 + .04 本跳 + .04 剩餘 > .1；直接到目的地則 .08 可行。
        self.assertEqual(env.check_action(2), "delay_lower_bound_exceeded")
        self.assertEqual(env.check_action(3), "")

    def test_initial_impossibility_and_all_masked_terminal(self):
        for topology, q, category, reason in (
            (FixedConstellation(), request(delay=.001), "initial_delay_infeasible", "delay_lower_bound_exceeded"),
            (FixedConstellation(capacity=10), request(), "initial_bandwidth_infeasible", "insufficient_bandwidth"),
            (FixedConstellation(edges=()), request(), "initial_no_physical_route", "no_physical_neighbors")):
            env = SFCOrchestrationEnv(topology)
            roll = RolloutCollector(env, agent()).collect(q)
            final = roll.transitions[-1]
            self.assertIsNone(final.action)
            self.assertTrue(final.terminated)
            self.assertEqual(final.next_value.item(), 0)
            self.assertEqual(final.info["failure_category"], category)
            self.assertIn(reason, final.info["no_action_reasons"])
            self.assertEqual(env.resources.metrics()["cpu_reserved"], 0)
            self.assertEqual(env.resources.metrics()["bandwidth_reserved_mhz"], 0)

    def test_route_feasibility_does_not_prove_cpu_feasibility(self):
        env = SFCOrchestrationEnv(FixedConstellation(cpu=50))
        roll = RolloutCollector(env, agent()).collect(request())
        info = roll.transitions[-1].info
        self.assertTrue(info["initial_diagnostics"]["route_feasible"])
        self.assertFalse(roll.accepted)
        self.assertEqual(info["failure_category"], "initial_endpoint_cpu_infeasible")

    def test_multiple_rejection_reasons(self):
        env = SFCOrchestrationEnv(FixedConstellation())
        env.reset(request=request())
        env.step(1)
        env.request = replace(env.request, bandwidth_mhz=250)
        self.assertEqual(set(env.action_reasons(0)), {"path_contains_cycle", "insufficient_bandwidth"})

    def test_matching_beats_independent_nearest_conflict(self):
        costs = [[1.,2.,9.],[1.,8.,9.],[9.,1.,2.]]
        assignment = minimum_distance_matching(costs)
        self.assertEqual(len(set(assignment)), 3)
        self.assertEqual(sum(costs[i][j] for i,j in enumerate(assignment)),
                         min(sum(costs[i][j] for i,j in enumerate(p)) for p in permutations(range(3))))

    def test_25_nodes_nominal_four_neighbors_and_los(self):
        constellation = LEOConstellation()
        self.assertEqual((constellation.config.satellites, constellation.config.planes), (25,5))
        self.assertEqual(constellation.config.inclination_deg, 85)
        audit = audit_topology(constellation)
        self.assertTrue(audit["passed"])
        self.assertTrue(all(row["nominal_links"] == 50 for row in audit["snapshots"]))
        for time in (0, 123, constellation.orbital_period_s/4):
            snap = constellation.snapshot(time)
            for n, sat in snap.satellites.items():
                nominal = [b if a == n else a for a,b in snap.nominal_links if n in (a,b)]
                self.assertEqual(len(nominal), 4)
                self.assertEqual(sum(snap.satellites[j].plane == sat.plane for j in nominal), 2)
                self.assertEqual({snap.satellites[j].plane for j in nominal if snap.satellites[j].plane != sat.plane},
                                 {(sat.plane-1)%5,(sat.plane+1)%5})
            edge = next(iter(snap.links))
            disabled = constellation.snapshot(time, disabled_links=[edge])
            self.assertIn(edge, disabled.nominal_links)
            self.assertNotIn(edge, disabled.links)

    def test_service_bounds_and_single_distribution(self):
        workload = RequestGenerator().generate(600,25)
        self.assertTrue(all(3 <= len(q.vnfs) <= 5 and not hasattr(q,"service_type") for q in workload))
        for count in (2,6):
            with self.assertRaisesRegex(ValueError, "3 to 5"):
                replace(workload[0], vnfs=tuple(VNF(str(i),50) for i in range(count)))
        with self.assertRaises(ValueError):
            RequestConfig(vnf_count_max=6)

    def test_endpoint_modes_and_diagnostics(self):
        snapshot = FixedConstellation().snapshot(0)
        satellites = dict(snapshot.satellites)
        satellites[0] = replace(satellites[0], cpu_capacity=50)
        snapshot = NetworkSnapshot(0, satellites, snapshot.links)
        manager = ResourceManager(snapshot)
        fixed = embed_vnfs(manager, request(), (0,1,2,3), "fixed")
        neighbor = embed_vnfs(manager, request(), (0,1,2,3), "neighbor_allowed")
        self.assertIsNone(fixed.placements[0])
        self.assertEqual(fixed.failure_reasons[0], "insufficient_cpu_at_allowed_endpoint")
        self.assertEqual(neighbor.placements[0], 1)
        diagnostic = arrival_diagnostics(manager, request(), "neighbor_allowed")
        self.assertTrue(diagnostic["endpoint_cpu_feasible"])
        self.assertEqual(diagnostic["source_endpoint_cpu_demand"], 100)
        self.assertIn("0", diagnostic["source_endpoint_available_cpu"])

    def test_checkpoint_node_mismatch_fails_before_loading_weights(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/"old.pt"
            save_policy(path, ActorCriticAgent(20))
            with self.assertRaisesRegex(ValueError, "20 satellites; expected 25"):
                load_policy(path, expected_num_nodes=25)
            torch.save({"version":1,"num_nodes":25}, path)
            with self.assertRaisesRegex(ValueError, "feature schema"):
                load_policy(path, expected_num_nodes=25)

    def test_normalized_migration_denominators(self):
        q = replace(request(), duration_s=20)
        result = run_dynamic(SimulationConfig(horizon_s=20,topology_step_s=5), [q],
            lambda manager,q: Allocation((0,1,3),(0,1,3)), topology=ChangingTopology(), repair=True)
        self.assertEqual(result["affected_services"],1)
        self.assertEqual(result["repair_success_rate"],1)
        self.assertEqual(result["vnf_migrations_per_accepted_sfc"],1)
        self.assertEqual(result["vnf_migrations_per_affected_sfc"],1)
        no_repair = run_dynamic(SimulationConfig(horizon_s=20,topology_step_s=5), [q],
            lambda manager,q: Allocation((0,1,3),(0,1,3)), topology=ChangingTopology(), repair=False)
        self.assertEqual(no_repair["affected_services"], 1)  # 同服務多事件受影響只計一次。
        self.assertEqual(no_repair["repair_attempts"], 0)
        empty = run_dynamic(SimulationConfig(horizon_s=1), [], lambda manager,q: None)
        self.assertIsNone(empty["repair_success_rate"])
        self.assertIsNone(empty["vnf_migrations_per_accepted_sfc"])
        self.assertIsNone(empty["vnf_migrations_per_affected_sfc"])
