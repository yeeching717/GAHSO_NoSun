"""命令列入口：執行事件模擬，或觀察逐跳 MDP 的狀態與獎勵。"""
import argparse
import json
from pathlib import Path
from .simulation import Simulation, SimulationConfig
from .mdp_demo import run_mdp_demo


def main():
    parser = argparse.ArgumentParser(description="LEO/SFC environment and GAHSO MDP")
    parser.add_argument("--config", type=Path, default=Path("config/default.json"))
    parser.add_argument("--mode", choices=("baseline", "mdp-demo", "gat-demo", "actor-critic-demo"), default="baseline")
    parser.add_argument("--output", type=Path, help="Optional complete JSON report")
    args = parser.parse_args()
    # 設定統一解析；JSON 不允許註解，各參數說明位於 dataclass 及 README。
    config = SimulationConfig.from_dict(json.loads(args.config.read_text(encoding="utf-8")))
    if args.mode == "actor-critic-demo":
        from .actor_critic_demo import run_actor_critic_demo
        report = run_actor_critic_demo(config)
        summary = {key: value for key, value in report.items() if key != "trajectory"}
    elif args.mode == "gat-demo":
        # PyTorch 僅在 GAT 模式載入；基準及 MDP 模擬保留標準函式庫相依性。
        from .gat_demo import run_gat_demo
        report = run_gat_demo(config)
        summary = {key: value for key, value in report.items() if key not in ("embeddings", "attention")}
    elif args.mode == "mdp-demo":
        report = run_mdp_demo(config)
        summary = {key: report[key] for key in ("mode", "policy", "before_expiry", "after_expiry")}
        summary["steps"] = len(report["trajectory"])
        summary["rewards"] = [step["reward"] for step in report["trajectory"]]
    else:
        report = Simulation(config).run()
        summary = {key: report[key] for key in ("policy", "generated_requests", "at_horizon", "final")}
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if args.output:
        # 完整報告含候選狀態與各步回饋，便於之後追查訓練問題。
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Report: {args.output.resolve()}")


if __name__ == "__main__":
    main()
