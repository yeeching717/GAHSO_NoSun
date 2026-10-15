import math
import unittest
from dataclasses import replace
from gahso.network import (ConstellationConfig, GroundTerminal, LEOConstellation,
                           Link, NetworkSnapshot, Satellite, edge_key, line_of_sight)
from gahso.resources import Allocation, ResourceManager, ServiceStatus
from gahso.services import RequestConfig, RequestGenerator, SFCRequest, VNF
from gahso.baseline import shortest_delay_greedy
from gahso.simulation import Simulation, SimulationConfig


def snapshot(time_s=0.0, edges=((0, 1), (1, 2), (2, 3), (0, 2), (1, 3)), delay_s=0.01):
    satellites = {n: Satellite(n, 0, n, (8000.0, n * 10.0, 0.0), 500.0) for n in range(4)}
    links = {edge_key(a, b): Link(*edge_key(a, b), 299000 * delay_s, delay_s, 200.0, "test")
             for a, b in edges}
    return NetworkSnapshot(time_s, satellites, links)


def request(request_id="q", cpu=(100, 100, 100), bandwidth=50, arrival=0, duration=10, delay=0.1):
    return SFCRequest(request_id, 0, 3, tuple(VNF(str(i), value) for i, value in enumerate(cpu)),
                      bandwidth, delay, arrival, duration)


class GeometryTests(unittest.TestCase):
    def setUp(self):
        self.constellation = LEOConstellation()

    def test_paper_defaults_and_four_transceiver_bound(self):
        snap = self.constellation.snapshot(123.0)
        self.assertEqual(len(snap.satellites), 25)
        for n, satellite in snap.satellites.items():
            self.assertAlmostEqual(math.sqrt(sum(x * x for x in satellite.position_km)), 8371.0, places=7)
            self.assertTrue(400 <= satellite.cpu_capacity <= 600)
            self.assertLessEqual(len(snap.neighbors(n)), 4)
            intra = [k for k in snap.neighbors(n) if snap.satellites[k].plane == satellite.plane]
            self.assertEqual(len(intra), 2)
        for (a, b), link in snap.links.items():
            self.assertIn(a, snap.neighbors(b))
            self.assertAlmostEqual(link.delay_s, link.distance_km / 299000)
            self.assertEqual(link.capacity_mhz, 200)
            self.assertTrue(line_of_sight(snap.satellites[a].position_km,
                                          snap.satellites[b].position_km, 6371))

    def test_orbital_period_and_motion(self):
        start = self.constellation.snapshot(0)
        later = self.constellation.snapshot(100)
        full = self.constellation.snapshot(self.constellation.orbital_period_s)
        self.assertNotEqual(start.satellites[0].position_km, later.satellites[0].position_km)
        self.assertNotEqual(start.links[(0, 5)].delay_s, later.links[(0, 5)].delay_s)
        for n in start.satellites:
            for a, b in zip(start.satellites[n].position_km, full.satellites[n].position_km):
                self.assertAlmostEqual(a, b, places=6)

    def test_geometry_and_capacity_are_reproducible(self):
        self.assertEqual(self.constellation.snapshot(27), LEOConstellation().snapshot(27))

    def test_supported_paper_scales(self):
        for satellites, planes in ((20, 4), (25, 5), (100, 10), (200, 10)):
            with self.subTest(satellites=satellites):
                snap = LEOConstellation(ConstellationConfig(satellites=satellites, planes=planes)).snapshot(31)
                self.assertEqual(len(snap.satellites), satellites)
                self.assertTrue(all(len(snap.neighbors(n)) <= 4 for n in snap.satellites))

    def test_disabled_link_is_removed(self):
        key = next(iter(self.constellation.snapshot(0).links))
        self.assertNotIn(key, self.constellation.snapshot(0, disabled_links=[key]).links)

    def test_earth_occlusion(self):
        self.assertFalse(line_of_sight((8371, 0, 0), (-8371, 0, 0), 6371))
        self.assertTrue(line_of_sight((8371, 0, 0), (8371, 10, 0), 6371))

    def test_terminal_selects_visible_nearest_satellite(self):
        self.assertEqual(GroundTerminal(0, 0).nearest_visible(self.constellation.snapshot(0)), 0)

    def test_invalid_config_and_time(self):
        for kwargs in ({"satellites": 101}, {"cpu_min": -1}, {"phase_factor": 5},
                       {"inclination_deg": float("nan")}, {"planes": 2}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                ConstellationConfig(**kwargs)
        with self.assertRaises(ValueError):
            self.constellation.snapshot(-1)


class WorkloadTests(unittest.TestCase):
    def test_reproducible_arrivals_and_paper_demand_bounds(self):
        generator = RequestGenerator()
        workload = generator.generate(600, 100)
        self.assertEqual(workload, generator.generate(600, 100))
        self.assertGreater(len(workload), 0)
        self.assertEqual(len({q.id for q in workload}), len(workload))
        self.assertEqual([q.arrival_s for q in workload], sorted(q.arrival_s for q in workload))
        for q in workload:
            self.assertLess(q.arrival_s, 600)
            self.assertGreater(q.expires_s, q.arrival_s)
            self.assertNotEqual(q.source, q.destination)
            self.assertTrue(20 <= q.bandwidth_mhz <= 80)
            self.assertTrue(3 <= len(q.vnfs) <= 5)
            self.assertTrue(all(50 <= v.cpu <= 200 for v in q.vnfs))

    def test_arrival_rate_and_single_service_distribution(self):
        cfg = RequestConfig(arrival_rate_per_slot=3)
        workload = RequestGenerator(cfg).generate(10000, 100)
        self.assertTrue(2700 < len(workload) < 3300)
        self.assertTrue(all(not hasattr(q, "service_type") for q in workload))
        self.assertGreater(max(q.bandwidth_mhz for q in workload), 79)

    def test_zero_arrivals(self):
        self.assertEqual(RequestGenerator(RequestConfig(arrival_rate_per_slot=0)).generate(10, 100), [])

    def test_input_validation(self):
        for kwargs in ({"arrival_rate_per_slot": -1}, {"mean_duration_s": 0},
                       {"vnf_count_min": 1}, {"cpu_min": 201}, {"max_delay_s": float("inf")}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                RequestConfig(**kwargs)
        with self.assertRaises(ValueError):
            request(cpu=(100,))
        with self.assertRaises(ValueError):
            VNF("bad", -1)


class ResourceTests(unittest.TestCase):
    def setUp(self):
        self.manager = ResourceManager(snapshot())
        self.plan = Allocation((0, 1, 3), (0, 1, 3))

    def test_admission_reserves_exact_resources(self):
        self.assertTrue(self.manager.admit(request(), self.plan).accepted)
        self.assertEqual(self.manager.cpu_available(0), 400)
        self.assertEqual(self.manager.cpu_available(1), 400)
        self.assertEqual(self.manager.cpu_available(2), 500)
        self.assertEqual(self.manager.bandwidth_available(0, 1), 150)
        self.assertEqual(self.manager.bandwidth_available(1, 0), 150)
        self.assertEqual(self.manager.metrics()["cpu_reserved"], 300)
        self.manager.assert_invariants()

    def test_expiry_returns_all_resources_once(self):
        self.manager.admit(request(), self.plan)
        self.manager.update(snapshot(9))
        self.assertEqual(self.manager.records["q"].status, ServiceStatus.ACTIVE)
        self.manager.update(snapshot(10))
        self.assertEqual(self.manager.records["q"].status, ServiceStatus.EXPIRED)
        self.assertEqual(self.manager.metrics()["cpu_reserved"], 0)
        self.assertEqual(self.manager.metrics()["bandwidth_reserved_mhz"], 0)
        self.assertFalse(self.manager.release("q"))
        self.manager.assert_invariants()

    def test_shared_link_and_independent_expiry(self):
        self.manager.admit(request("a", duration=5), self.plan)
        self.manager.admit(request("b", duration=10), self.plan)
        self.manager.update(snapshot(5))
        self.assertEqual(self.manager.bandwidth_available(0, 1), 150)
        self.assertEqual(self.manager.cpu_available(1), 400)
        self.manager.update(snapshot(10))
        self.assertEqual(self.manager.bandwidth_available(0, 1), 200)
        self.manager.assert_invariants()

    def test_cpu_failure_is_atomic(self):
        bad = Allocation((0, 1, 3), (0, 0, 3))
        result = self.manager.admit(request(cpu=(300, 300, 100)), bad)
        self.assertEqual(result.reason, "insufficient_cpu")
        self.assertEqual(self.manager.metrics()["cpu_reserved"], 0)
        self.assertEqual(self.manager.metrics()["bandwidth_reserved_mhz"], 0)
        self.manager.assert_invariants()

    def test_bandwidth_failure_does_not_consume_cpu(self):
        self.manager.admit(request("a", bandwidth=120), self.plan)
        before = self.manager.cpu_available(0)
        result = self.manager.admit(request("b", bandwidth=120), self.plan)
        self.assertEqual(result.reason, "insufficient_bandwidth")
        self.assertEqual(self.manager.cpu_available(0), before)
        self.assertEqual(self.manager.bandwidth_available(0, 1), 80)
        self.manager.assert_invariants()

    def test_duplicate_admission_and_double_release(self):
        self.manager.admit(request(), self.plan)
        self.assertEqual(self.manager.admit(request(), self.plan).reason, "duplicate_request_id")
        self.assertEqual(self.manager.metrics()["cpu_reserved"], 300)
        self.assertTrue(self.manager.release("q"))
        self.assertFalse(self.manager.release("q"))
        self.manager.assert_invariants()

    def test_endpoint_order_cycle_and_delay_validation(self):
        cases = [
            (request(), Allocation((0, 1, 3), (1, 1, 3)), "invalid_vnf_endpoints"),
            (request(cpu=(50, 50, 50, 50)), Allocation((0, 1, 2, 3), (0, 2, 1, 3)), "vnf_order_violation"),
            (request(), Allocation((0, 1, 0, 3), (0, 1, 3)), "path_contains_cycle"),
            (request(delay=0.005), self.plan, "delay_exceeded"),
            (request(), Allocation((0, 3), (0, 0, 3)), "link_unavailable"),
            (request(arrival=1), self.plan, "request_not_arrived"),
        ]
        for q, plan, reason in cases:
            with self.subTest(reason=reason):
                manager = ResourceManager(snapshot())
                self.assertEqual(manager.admit(q, plan).reason, reason)
                self.assertEqual(manager.metrics()["cpu_reserved"], 0)
                manager.assert_invariants()

    def test_colocated_vnfs_and_same_endpoint_request(self):
        q = replace(request(cpu=(100, 100, 100)), destination=0)
        self.assertTrue(self.manager.admit(q, Allocation((0,), (0, 0, 0))).accepted)
        self.assertEqual(self.manager.cpu_available(0), 200)
        self.assertEqual(self.manager.metrics()["bandwidth_reserved_mhz"], 0)

    def test_interruption_restoration_and_expiry_on_missing_link(self):
        self.manager.admit(request(), self.plan)
        broken = ((0, 2), (2, 3))
        self.manager.update(snapshot(1, broken))
        self.assertEqual(self.manager.pending_migrations(), ["q"])
        self.assertEqual(self.manager.metrics()["cpu_reserved"], 300)
        self.manager.update(snapshot(2, broken))
        self.assertEqual(self.manager.records["q"].interruption_count, 1)
        self.manager.update(snapshot(3))
        self.assertEqual(self.manager.pending_migrations(), [])
        self.assertEqual(self.manager.bandwidth_available(0, 1), 150)
        self.manager.update(snapshot(4, broken))
        self.manager.update(snapshot(10, broken))
        self.assertEqual(self.manager.metrics()["bandwidth_reserved_mhz"], 0)
        self.assertEqual(self.manager.records["q"].interruption_count, 2)
        self.manager.assert_invariants()

    def test_topology_delay_violation_detected(self):
        self.manager.admit(request(delay=0.025), self.plan)
        self.manager.update(snapshot(1, delay_s=0.02))
        self.assertEqual(self.manager.records["q"].reason, "delay_exceeded")

    def test_atomic_reallocation_credits_existing_reservations(self):
        self.manager.admit(request(cpu=(450, 450, 450)), self.plan)
        self.manager.update(snapshot(1, ((0, 2), (2, 3))))
        self.assertTrue(self.manager.replace_allocation("q", Allocation((0, 2, 3), (0, 2, 3))).accepted)
        self.assertEqual(self.manager.cpu_available(1), 500)
        self.assertEqual(self.manager.cpu_available(2), 50)
        self.assertEqual(self.manager.pending_migrations(), [])
        self.assertEqual(self.manager.metrics()["cpu_reserved"], 1350)
        self.manager.assert_invariants()

    def test_failed_reallocation_keeps_old_reservations(self):
        self.manager.admit(request(cpu=(300, 300, 300)), self.plan)
        bad = Allocation((0, 2, 3), (0, 0, 3))
        self.assertEqual(self.manager.replace_allocation("q", bad).reason, "insufficient_cpu")
        self.assertEqual(self.manager.records["q"].allocation, self.plan)
        self.assertEqual(self.manager.bandwidth_available(0, 1), 150)
        self.assertEqual(self.manager.bandwidth_available(0, 2), 200)
        self.manager.assert_invariants()

    def test_time_cannot_go_backwards(self):
        self.manager.update(snapshot(3))
        with self.assertRaises(ValueError):
            self.manager.update(snapshot(2))

    def test_jain_fairness_matches_manual_loads(self):
        self.assertEqual(self.manager.metrics()["load_fairness"], 1)
        self.manager.admit(request(), self.plan)
        # Loads: node 0=150, node 1=200, node 2=0, node 3=150.
        expected = 500 ** 2 / (4 * (150 ** 2 + 200 ** 2 + 150 ** 2))
        self.assertAlmostEqual(self.manager.metrics()["load_fairness"], expected)


class IntegrationTests(unittest.TestCase):
    def test_baseline_policy_produces_a_valid_allocation(self):
        manager = ResourceManager(snapshot())
        q = request()
        plan = shortest_delay_greedy(manager, q)
        self.assertIsNotNone(plan)
        self.assertTrue(manager.admit(q, plan).accepted)
        manager.assert_invariants()

    def test_seeded_simulation_drains_and_reproduces(self):
        cfg = SimulationConfig(horizon_s=40, topology_step_s=10,
                               requests=RequestConfig(mean_duration_s=15))
        first, second = Simulation(cfg).run(), Simulation(cfg).run()
        self.assertEqual(first, second)
        self.assertGreater(first["final"]["accepted"], 0)
        self.assertEqual(first["final"]["cpu_reserved"], 0)
        self.assertEqual(first["final"]["bandwidth_reserved_mhz"], 0)
        self.assertEqual(first["final"]["statuses"]["active"], 0)
        self.assertEqual(first["final"]["statuses"]["interrupted"], 0)
        self.assertEqual(first["at_horizon"]["time_s"], 40)

    def test_same_time_expiry_precedes_new_admission(self):
        cfg = SimulationConfig(horizon_s=10, topology_step_s=10, drain=True)
        a = SFCRequest("a", 0, 1, (VNF("s", 400), VNF("m", 10), VNF("d", 400)), 150, 1, 0, 5)
        b = replace(a, id="b", arrival_s=5)
        report = Simulation(cfg).run([a, b])
        self.assertEqual(report["final"]["accepted"], 2)
        events = [e["event"] for e in report["events"] if e["time_s"] == 5]
        self.assertEqual(events[:2], ["expired", "accepted"])

    def test_no_drain_preserves_unexpired_reservations(self):
        cfg = SimulationConfig(horizon_s=1, topology_step_s=1, drain=False)
        q = SFCRequest("q", 0, 1, (VNF("s", 50), VNF("m", 50), VNF("d", 50)), 20, 1, 0, 100)
        report = Simulation(cfg).run([q])
        self.assertEqual(report["final"]["statuses"]["active"], 1)
        self.assertEqual(report["final"]["cpu_reserved"], 150)

    def test_empty_workload_and_single_use(self):
        sim = Simulation(SimulationConfig(horizon_s=1))
        report = sim.run([])
        self.assertEqual(report["final"]["requests"], 0)
        self.assertEqual(report["final"]["load_fairness"], 1)
        with self.assertRaises(RuntimeError):
            sim.run([])

    def test_duplicate_workload_ids_rejected(self):
        with self.assertRaises(ValueError):
            Simulation().run([request(), request()])


if __name__ == "__main__":
    unittest.main()
