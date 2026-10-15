"""GAT 特徵、公式、梯度與圖結構不變性的測試；PyTorch 為此階段必要相依。"""
import copy
import math
import unittest
from dataclasses import replace
import torch
from gahso import SFCOrchestrationEnv, SFCRequest, VNF
from gahso.features import GATFeatureBuilder
from gahso.gat import EdgeAwareGATLayer, GAHSOGAT
from gahso.gat_config import GATConfig, GATFeatureConfig
from gahso.network import ConstellationConfig, LEOConstellation
from gahso.simulation import SimulationConfig
from helpers import check_gat_gradients


def setUpModule():
    global previous_threads
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)


def tearDownModule():
    torch.set_num_threads(previous_threads)


def make_observation():
    env = SFCOrchestrationEnv()
    q = SFCRequest("gat-test", 0, 2, (VNF("a", 50), VNF("b", 150), VNF("c", 75)),
                   30, 0.1, 0, 30)
    obs, _ = env.reset(request=q)
    return env, obs


class FeatureTests(unittest.TestCase):
    def test_dimensions_values_and_metadata(self):
        _, obs = make_observation()
        graph = GATFeatureBuilder().build(obs)
        self.assertEqual(tuple(graph.x.shape), (25, 25))
        self.assertEqual(tuple(graph.edge_attr.shape), (len(obs["edge_features"]), 2))
        self.assertEqual(tuple(graph.edge_index.shape), (2, len(obs["edge_features"])))
        self.assertEqual(graph.edge_index.dtype, torch.long)
        self.assertEqual(graph.action_mask.dtype, torch.bool)
        self.assertEqual(graph.x.dtype, torch.float32)
        self.assertEqual(len(graph.node_feature_names), 25)
        self.assertEqual(tuple(graph.request_features.shape), (15,))
        self.assertAlmostEqual(graph.request_features[0].item(), 30 / 200)
        self.assertAlmostEqual(graph.request_features[2].item(), 3 / 5)
        torch.testing.assert_close(graph.request_features[5:8], torch.tensor([0.25, 0.75, 0.375]))
        torch.testing.assert_close(graph.request_features[10:15], torch.tensor([1., 1., 1., 0., 0.]))
        torch.testing.assert_close(graph.x[:, 10:], graph.request_features.expand(25, -1))

    def test_absolute_cpu_distinguishes_equal_utilization(self):
        _, obs = make_observation()
        graph = GATFeatureBuilder().build(obs)
        self.assertEqual(graph.x[0, 0].item(), 1)
        self.assertEqual(graph.x[1, 0].item(), 1)
        column = graph.node_feature_names.index("cpu_remaining_scaled")
        self.assertNotEqual(graph.x[0, column].item(), graph.x[1, column].item())
        torch.testing.assert_close(graph.x[:, column], torch.tensor(obs["cpu_remaining"]) / 200)

    def test_vnf_order_bandwidth_delay_are_not_lost(self):
        _, obs = make_observation()
        builder = GATFeatureBuilder()
        original = builder.build(obs)
        for field in ("vnfs", "bandwidth_mhz", "max_delay_s"):
            changed = copy.deepcopy(obs)
            if field == "vnfs":
                changed["request"][field] = tuple(reversed(changed["request"][field]))
            else:
                changed["request"][field] *= 2
            self.assertFalse(torch.equal(original.x, builder.build(changed).x))

    def test_step_changes_position_order_and_candidate_bandwidth(self):
        env, initial = make_observation()
        next_state, *_ = env.step(1)
        builder = GATFeatureBuilder()
        first, second = builder.build(initial), builder.build(next_state)
        path_column = second.node_feature_names.index("path_order_scaled")
        self.assertAlmostEqual(second.x[1, path_column].item(), 1 / 25)
        self.assertEqual(second.position.argmax().item(), 1)
        self.assertFalse(torch.equal(first.edge_attr, second.edge_attr))
        self.assertAlmostEqual(second.request_features[4].item(), 24 / 25)
        self.assertEqual(env.resources.metrics()["bandwidth_reserved_mhz"], 0)

    def test_padding_limit_rejects_instead_of_truncating(self):
        _, obs = make_observation()
        obs["request"]["vnfs"] = tuple({"name": str(i), "cpu": 50} for i in range(9))
        with self.assertRaisesRegex(ValueError, "VNF count"):
            GATFeatureBuilder().build(obs)
        with self.assertRaises(ValueError):
            GATFeatureConfig(max_vnfs=9)

    def test_idle_and_edgeless_state_shapes(self):
        env = SFCOrchestrationEnv()
        idle, _ = env.reset()
        idle["edge_index"] = ((), ())
        idle["edge_features"] = ()
        graph = GATFeatureBuilder().build(idle)
        self.assertEqual(tuple(graph.edge_index.shape), (2, 0))
        self.assertEqual(tuple(graph.edge_attr.shape), (0, 2))
        self.assertEqual(graph.request_features.sum().item(), 0)
        self.assertEqual(graph.position.sum().item(), 0)

    def test_negative_remaining_delay_preserved(self):
        _, obs = make_observation()
        obs["accumulated_delay_s"] = 0.2
        graph = GATFeatureBuilder().build(obs)
        self.assertAlmostEqual(graph.request_features[3].item(), -1)

    def test_dtype_transfer_keeps_indices_and_masks(self):
        _, obs = make_observation()
        graph = GATFeatureBuilder().build(obs).to("cpu", dtype=torch.float64)
        self.assertEqual(graph.x.dtype, torch.float64)
        self.assertEqual(graph.edge_attr.dtype, torch.float64)
        self.assertEqual(graph.edge_index.dtype, torch.long)
        self.assertEqual(graph.action_mask.dtype, torch.bool)
        with self.assertRaises(ValueError):
            graph.to(dtype=torch.long)

    def test_builder_does_not_mutate_or_alias_observation(self):
        env, obs = make_observation()
        backup = copy.deepcopy(obs)
        graph = GATFeatureBuilder().build(obs)
        graph.x.zero_()
        self.assertEqual(obs, backup)
        self.assertEqual(env.resources.metrics()["cpu_reserved"], 0)

    def test_misaligned_nonfinite_and_wrong_schema_inputs_rejected(self):
        _, obs = make_observation()
        mutations = [
            ("node_ids", tuple(range(19))),
            ("edge_features", ()),
            ("node_feature_names", ("wrong",)),
            ("action_mask", (True,) * 19),
            ("position", (0,) * 25),
            ("path", (0, 0)),
            ("step_count", -1),
        ]
        for key, value in mutations:
            with self.subTest(key=key), self.assertRaises(ValueError):
                GATFeatureBuilder().build(dict(obs, **{key: value}))
        bad = copy.deepcopy(obs)
        bad["node_features"] = ((float("nan"),) + obs["node_features"][0][1:],) + obs["node_features"][1:]
        with self.assertRaises(ValueError):
            GATFeatureBuilder().build(bad)
        with self.assertRaises(ValueError):
            GATFeatureBuilder().build(dict(obs, edge_index=((99,), (0,)), edge_features=((1, 1),)))

    def test_ids_are_metadata_not_node_row_indices(self):
        _, obs = make_observation()
        renamed = dict(obs)
        renamed["node_ids"] = tuple(n + 100 for n in obs["node_ids"])
        renamed["path"] = (100,)
        renamed["request"] = dict(obs["request"], source=100, destination=102)
        a, b = GATFeatureBuilder().build(obs), GATFeatureBuilder().build(renamed)
        torch.testing.assert_close(a.x, b.x)
        torch.testing.assert_close(a.edge_index, b.edge_index)


class AttentionTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(19)

    def test_formula_matches_independent_hand_calculation(self):
        # 1→0、2→0；每個節點另有模型自迴圈。手算 a=[1,2,3]，W_x=W_e=1。
        layer = EdgeAwareGATLayer(1, 1, 1).double()
        with torch.no_grad():
            layer.node_projection.weight.fill_(1)
            layer.edge_projection.weight.fill_(1)
            layer.attention.copy_(torch.tensor([[1., 2., 3.]], dtype=torch.float64))
        x = torch.tensor([[1.], [2.], [4.]], dtype=torch.float64)
        edges = torch.tensor([[1, 2], [0, 0]])
        attrs = torch.tensor([[0.1], [0.4]], dtype=torch.float64)
        output, attention = layer(x, edges, attrs, return_attention=True)
        scores = [5.3, 10.2, 3.0]
        weights = [math.exp(v - max(scores)) for v in scores]
        expected = sum(w * v for w, v in zip(weights, [2, 4, 1])) / sum(weights)
        self.assertAlmostEqual(output[0, 0].item(), expected, places=12)
        self.assertEqual(output[1, 0].item(), 2)
        self.assertEqual(output[2, 0].item(), 4)
        self.assertEqual(tuple(attention.alpha.shape), (5, 1))

    def test_edge_features_change_attention_and_output(self):
        layer = EdgeAwareGATLayer(1, 1, 1)
        with torch.no_grad():
            layer.node_projection.weight.fill_(1)
            layer.edge_projection.weight.fill_(1)
            layer.attention.copy_(torch.tensor([[0., 0., 1.]]))
        x = torch.tensor([[1.], [2.], [4.]])
        edges = torch.tensor([[1, 2], [0, 0]])
        first, a = layer(x, edges, torch.zeros(2, 1), return_attention=True)
        second, b = layer(x, edges, torch.tensor([[0.], [4.]]), return_attention=True)
        self.assertFalse(torch.allclose(a.alpha, b.alpha))
        self.assertGreater(second[0, 0].item(), first[0, 0].item())

    def test_attention_sums_to_one_per_receiver_and_head(self):
        layer = EdgeAwareGATLayer(3, 2, 4, heads=3)
        edges = torch.tensor([[0, 1, 2, 2], [1, 0, 0, 1]])
        out, attention = layer(torch.randn(4, 3), edges, torch.randn(4, 2), return_attention=True)
        totals = torch.zeros(4, 3).index_add_(0, attention.edge_index[1], attention.alpha)
        torch.testing.assert_close(totals, torch.ones(4, 3))
        self.assertEqual(tuple(out.shape), (4, 4))
        self.assertTrue(bool((attention.alpha >= 0).all()))

    def test_self_loops_once_and_isolated_nodes(self):
        layer = EdgeAwareGATLayer(2, 2, 3)
        x = torch.randn(3, 2)
        edges = torch.tensor([[0, 0], [0, 1]])
        attrs = torch.randn(2, 2)
        out, attention = layer(x, edges, attrs, return_attention=True)
        self.assertEqual(int((attention.edge_index[0] == attention.edge_index[1]).sum()), 3)
        torch.testing.assert_close(out[2], layer.node_projection(x)[2])
        self.assertEqual(attention.alpha[attention.edge_index[1] == 2].item(), 1)

    def test_no_edges_still_projects_each_node(self):
        layer = EdgeAwareGATLayer(2, 2, 3)
        x = torch.randn(5, 2)
        out = layer(x, torch.empty(2, 0, dtype=torch.long), torch.empty(0, 2))
        torch.testing.assert_close(out, layer.node_projection(x))

    def test_non_neighbor_does_not_send_messages(self):
        layer = EdgeAwareGATLayer(2, 2, 3)
        edges = torch.tensor([[0, 1], [1, 0]])
        x, attrs = torch.randn(3, 2), torch.randn(2, 2)
        a = layer(x, edges, attrs)
        x[2] += 1000
        b = layer(x, edges, attrs)
        torch.testing.assert_close(a[:2], b[:2])
        self.assertFalse(torch.allclose(a[2], b[2]))

    def test_equivariance_under_node_permutation_and_edge_order(self):
        layer = EdgeAwareGATLayer(3, 2, 4, heads=2).double()
        x = torch.randn(4, 3, dtype=torch.float64)
        edges = torch.tensor([[0, 1, 2, 3, 1], [1, 2, 3, 0, 0]])
        attrs = torch.randn(5, 2, dtype=torch.float64)
        original = layer(x, edges, attrs)
        permutation = torch.tensor([2, 0, 3, 1])
        inverse = torch.argsort(permutation)
        edge_order = torch.tensor([4, 2, 0, 3, 1])
        permuted = layer(x[permutation], inverse[edges[:, edge_order]], attrs[edge_order])
        torch.testing.assert_close(permuted, original[permutation])

    def test_stable_softmax_with_large_finite_logits(self):
        layer = EdgeAwareGATLayer(2, 2, 2)
        with torch.no_grad():
            layer.node_projection.weight.fill_(1)
            layer.edge_projection.weight.fill_(1)
            layer.attention.fill_(1000)
        out, attention = layer(torch.tensor([[1., 2.], [100., 200.]]),
                               torch.tensor([[0, 1], [1, 0]]), torch.ones(2, 2), return_attention=True)
        self.assertTrue(bool(torch.isfinite(out).all()))
        self.assertTrue(bool(torch.isfinite(attention.alpha).all()))

    def test_autograd_matches_numerical_gradients(self):
        layer = EdgeAwareGATLayer(2, 2, 2).double()
        x = torch.randn(3, 2, dtype=torch.float64, requires_grad=True)
        attrs = torch.randn(4, 2, dtype=torch.float64, requires_grad=True)
        edges = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]])
        self.assertTrue(torch.autograd.gradcheck(
            lambda nodes, links: layer(nodes, edges, links), (x, attrs), fast_mode=True,
        ))

    def test_invalid_graphs_raise_clear_errors(self):
        layer = EdgeAwareGATLayer(2, 2, 3)
        x = torch.randn(3, 2)
        for edges, attrs in (
            (torch.tensor([[0], [3]]), torch.zeros(1, 2)),
            (torch.tensor([[0, 0], [1, 1]]), torch.zeros(2, 2)),
            (torch.tensor([[0.], [1.]]), torch.zeros(1, 2)),
            (torch.tensor([[0], [1]]), torch.zeros(0, 2)),
        ):
            with self.subTest(edges=edges), self.assertRaises(ValueError):
                layer(x, edges, attrs)


class EncoderTests(unittest.TestCase):
    def test_default_end_to_end_shape_and_all_parameter_gradients(self):
        _, obs = make_observation()
        graph = GATFeatureBuilder().build(obs)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(7)
            model = GAHSOGAT()
            result, attention = model(graph, return_attention=True)
            self.assertEqual(tuple(result.shape), (25, 128))
            self.assertEqual(len(attention), 2)
            result.square().mean().backward()
        for name, p in model.named_parameters():
            with self.subTest(parameter=name):
                self.assertIsNotNone(p.grad)
                self.assertTrue(bool(torch.isfinite(p.grad).all()))
                self.assertGreater(p.grad.abs().sum().item(), 0)

    def test_same_model_handles_other_constellation_sizes(self):
        model = GAHSOGAT()
        builder = GATFeatureBuilder()
        for satellites, planes in ((20, 4), (25, 5), (100, 10), (200, 10)):
            env = SFCOrchestrationEnv(LEOConstellation(ConstellationConfig(satellites=satellites, planes=planes)))
            state, _ = env.reset()
            with torch.no_grad():
                result = model(builder.build(state))
            self.assertEqual(tuple(result.shape), (satellites, 128))

    def test_eval_is_deterministic_and_state_dict_round_trip(self):
        _, obs = make_observation()
        graph = GATFeatureBuilder().build(obs)
        config = GATConfig(hidden_dim=16, num_layers=3, heads=2, dropout=0.5)
        model = GAHSOGAT(config=config).eval()
        other = GAHSOGAT(config=config).eval()
        other.load_state_dict(model.state_dict())
        torch.testing.assert_close(model(graph), model(graph))
        torch.testing.assert_close(model(graph), other(graph))

    def test_custom_feature_config_and_schema_mismatch(self):
        _, obs = make_observation()
        features = GATFeatureConfig(max_vnfs=5)
        graph = GATFeatureBuilder(features).build(obs)
        model = GAHSOGAT(features, GATConfig(hidden_dim=8, num_layers=1)).double()
        result = model(graph.to(dtype=torch.float64))
        self.assertEqual(tuple(result.shape), (25, 8))
        with self.assertRaises(ValueError):
            GAHSOGAT()(replace(graph, x=graph.x[:, :-1]))

    def test_fixture_reports_finite_backward_and_unchanged_resources(self):
        result = check_gat_gradients(SimulationConfig())
        self.assertEqual(result["output_shape"], [25, 128])
        self.assertFalse(result["trained"])
        self.assertTrue(result["resources_unchanged"])
        self.assertTrue(all(g["finite"] and g["norm"] > 0 for g in result["gradients"].values()))
        self.assertTrue(all(error < 1e-5 for error in result["attention_normalization_max_error"]))

    def test_config_rejects_invalid_dimensions_and_scales(self):
        for kwargs in ({"hidden_dim": 0}, {"heads": 0}, {"dropout": 1}, {"num_layers": 1.5}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                GATConfig(**kwargs)
        for kwargs in ({"max_vnfs": 1}, {"cpu_demand_scale": 0}, {"delay_scale_s": float("nan")}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                GATFeatureConfig(**kwargs)


if __name__ == "__main__":
    unittest.main()
