"""載入部署 checkpoint，在新工作負載上執行動態編排，不訓練模型。"""
import argparse
import json
from pathlib import Path
import torch
from .deployment import load_policy
from .dynamic import FaultTopology, run_dynamic
from .network import LEOConstellation
from .services import RequestGenerator
from .simulation import SimulationConfig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("config/default.json"))
    parser.add_argument("--output", type=Path, default=Path("results/deployment.json"))
    parser.add_argument("--failure-probability", type=float, default=0.)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--no-repair", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(1)
    config = SimulationConfig.from_dict(json.loads(args.config.read_text(encoding="utf-8")))
    policy = load_policy(args.checkpoint, expected_num_nodes=config.constellation.satellites)
    requests = RequestGenerator(config.requests).generate(config.horizon_s, config.constellation.satellites)
    topology = FaultTopology(LEOConstellation(config.constellation), probability=args.failure_probability,
                             slot_s=config.topology_step_s, seed=args.seed)
    report = run_dynamic(config, requests, policy, topology=topology, repair=not args.no_repair)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k not in
                      ("timeline", "decisions", "migration_events", "events", "final")}, indent=2))


if __name__ == "__main__":
    main()
