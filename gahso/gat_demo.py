"""以真實 20 顆衛星 MDP 狀態執行 GAT 前向傳播與反向傳播。"""
from dataclasses import asdict
import torch
from .features import GATFeatureBuilder
from .gat import GAHSOGAT
from .mdp import SFCOrchestrationEnv
from .network import LEOConstellation
from .services import SFCRequest, VNF


def run_gat_demo(config) -> dict:
    """隨機初始化模型的介面驗證；測試 loss 不代表 GAHSO 訓練目標。"""
    env = SFCOrchestrationEnv(LEOConstellation(config.constellation), config.mdp,
                             cpu_load_weight=config.cpu_load_weight,
                             bandwidth_load_weight=config.bandwidth_load_weight)
    request = SFCRequest("gat-demo", 0, 2,
                         (VNF("ingress", 50), VNF("process", 150), VNF("egress", 75)),
                         30, 0.1, 0, 30, "compute")
    observation, _ = env.reset(request=request)
    builder = GATFeatureBuilder(config.gat_features)
    graph = builder.build(observation)
    # 僅在示範中固定初始化種子，不改動外部訓練程序的全域 RNG 狀態。
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(7)
        model = GAHSOGAT(config.gat_features, config.gat)
        embeddings, attentions = model(graph, return_attention=True)
        loss = embeddings.square().mean()
        loss.backward()
    gradients = {
        name: {"finite": p.grad is not None and bool(torch.isfinite(p.grad).all()),
               "norm": float(p.grad.norm()) if p.grad is not None else None}
        for name, p in model.named_parameters()
    }
    attention_error = []
    for weights in attentions:
        sums = weights.alpha.new_zeros((graph.x.shape[0], config.gat.heads))
        sums.index_add_(0, weights.edge_index[1], weights.alpha.detach())
        attention_error.append(float((sums - 1).abs().max()))
    return {
        "mode": "gat_demo", "torch_version": torch.__version__,
        "device": str(graph.x.device), "trained": False,
        "feature_config": asdict(config.gat_features), "gat_config": asdict(config.gat),
        "node_input_shape": list(graph.x.shape), "edge_input_shape": list(graph.edge_attr.shape),
        "edge_index_shape": list(graph.edge_index.shape), "output_shape": list(embeddings.shape),
        "node_feature_names": graph.node_feature_names,
        "edge_feature_names": graph.edge_feature_names,
        "attention_normalization_max_error": attention_error,
        "backward_check_loss": float(loss.detach()),
        "gradients": gradients, "embeddings": embeddings.detach().tolist(),
        "attention": [{"edge_index": a.edge_index.tolist(), "alpha": a.alpha.detach().tolist()}
                      for a in attentions],
        "resources_unchanged": env.resources.metrics()["cpu_reserved"] == 0
                               and env.resources.metrics()["bandwidth_reserved_mhz"] == 0,
    }
