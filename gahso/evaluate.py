"""可重現實驗：訓練曲線、部署基準、動態修復，以及可重新載入的 Actor。"""
import argparse
import csv
import json
import platform
from functools import partial
from collections import Counter
from dataclasses import asdict, replace
from pathlib import Path
from statistics import mean, pstdev
from math import floor
from time import perf_counter
import torch
from .actor_critic import ActorCriticAgent
from .baseline import shortest_delay_greedy
from .deployment import DeploymentPolicy, FrozenTopology, save_policy, load_policy
from .dynamic import FaultTopology, run_dynamic
from .mdp import SFCOrchestrationEnv
from .migration import MigrationConfig
from .network import LEOConstellation
from .services import RequestGenerator
from .simulation import SimulationConfig
from .training import A2CTrainer, batch_request_losses
from .rollout import RolloutCollector
from .topology_audit import audit_topology


def workload(config, count, seed):
    if type(count) is not int or count <= 0 or config.requests.arrival_rate_per_slot <= 0:
        raise ValueError("Positive count and arrival rate required")
    horizon = config.horizon_s
    for _ in range(30):
        requests = RequestGenerator(replace(config.requests, seed=seed)).generate(horizon, config.constellation.satellites)
        if len(requests) >= count:
            requests = requests[:count]
            return requests, replace(config, horizon_s=max(config.horizon_s, requests[-1].arrival_s + 1))
        horizon *= 2
    raise ValueError("Could not generate requested workload")


def assess_convergence(history, window=5):
    """預先固定的有限樣本穩定性診斷，並非收斂證明。"""
    if len(history) < max(20, 2 * window):
        return {"status": "insufficient_epochs", "window": window, "minimum_epochs": max(20, 2*window)}
    old, new = history[-2*window:-window], history[-window:]
    def relative(key):
        a, b = mean(x[key] for x in old), mean(x[key] for x in new)
        # 縮放後 loss 常小於 1；以 1 當分母會把相對惡化誤判為穩定。
        return abs(b-a) / max(1e-8, abs(a))
    reward_change = relative("mean_reward")
    loss_change = relative("mean_critic_loss")
    success_change = abs(mean(x["success_rate"] for x in new) - mean(x["success_rate"] for x in old))
    improved = mean(x["mean_reward"] for x in new) > mean(x["mean_reward"] for x in history[:window])
    stable = reward_change <= .05 and loss_change <= .10 and success_change <= .05
    initial_success = mean(x["success_rate"] for x in history[:window])
    success_improved = mean(x["success_rate"] for x in new) > initial_success
    diagnostic_keys = ("mean_abs_td_error", "feasible_action_ratio")
    diagnostics_present = all(all(k in x for k in diagnostic_keys) for x in history)
    td_change = relative("mean_abs_td_error") if diagnostics_present else None
    td_improved = diagnostics_present and mean(x["mean_abs_td_error"] for x in new) < mean(
        x["mean_abs_td_error"] for x in history[:window])
    feasible_change = (mean(x["feasible_action_ratio"] for x in new) - mean(
        x["feasible_action_ratio"] for x in history[:window])) if diagnostics_present else None
    stable = stable and success_improved and td_improved and td_change <= .1
    return {"status": "stable_window_not_proof" if stable and improved else "not_converged",
            "window": window, "reward_relative_change": reward_change,
            "critic_loss_relative_change": loss_change, "success_rate_change": success_change,
            "reward_improved_from_initial_window": improved,
            "success_improved_from_initial_window": success_improved,
            "td_error_relative_change": td_change, "td_error_improved": td_improved,
            "relative_denominator_floor": 1e-8,
            "feasible_action_ratio_change": feasible_change,
            "feasible_action_ratio_improved": feasible_change is not None and feasible_change > 0,
            "note": "Feasible ratio depends on admitted load/path; it is diagnostic, not a convergence requirement.",
            "thresholds": {"reward": .05, "critic_loss": .10, "success_rate": .05}}


def assess_validation_convergence(validation_history, scenario="orbital"):
    """Use the fixed held-out workload; Actor loss sign is not a criterion."""
    if len(validation_history) < 4:
        return {"status": "insufficient_validation_points", "scenario": scenario}
    rows = [entry["scenarios"][scenario] for entry in validation_history]
    first, last = rows[:2], rows[-2:]
    avg = lambda group, key: mean(x[key] for x in group if x[key] is not None)
    conditional_change = avg(last, "conditional_success_rate") - avg(first, "conditional_success_rate")
    td_change = avg(last, "mean_abs_td_error") - avg(first, "mean_abs_td_error")
    critic_last = avg(last, "critic_loss")
    entropy_last = avg(last, "policy_entropy_mean")
    stable = conditional_change >= -.02 and td_change <= .02 and critic_last < 1e3 and entropy_last > .01
    return {"status": "stable_window_not_proof" if stable else "not_converged",
        "scenario": scenario, "validation_points": len(rows),
        "conditional_success_rate_change": conditional_change,
        "mean_abs_td_error_change": td_change, "last_critic_loss": critic_last,
        "last_policy_entropy": entropy_last,
        "criteria": {"conditional_success_drop_tolerance": .02,
                     "td_error_increase_tolerance": .02,
                     "critic_explosion_threshold": 1000, "entropy_collapse_threshold": .01}}


def training_batch(config, count, seed, epoch, mode, offset):
    """epoch_reset 用獨立流量；persistent 接續到達時間並避免請求 ID 重複。"""
    batch, _ = workload(config, count, seed)
    if mode == "persistent":
        batch = [replace(q, id=f"epoch-{epoch}-{q.id}", arrival_s=q.arrival_s + offset) for q in batch]
    return batch


def advance_training_time(env, arrival_s, topology_step_s):
    """與部署一樣處理拓樸 tick 及到期事件；請求仍在自己的到達時間決策。"""
    current = env.resources.time_s
    events = {r.request.expires_s for r in env.resources.records.values()
              if current < r.request.expires_s <= arrival_s and r.status.value in ("active", "interrupted")}
    tick = (floor(current / topology_step_s) + 1) * topology_step_s
    while tick <= arrival_s:
        events.add(tick)
        tick += topology_step_s
    for event in sorted(events):
        env.advance_time(event)


def _ratio(numerator, denominator):
    return numerator / denominator if denominator else None


def summarize_records(records):
    """Aggregate request diagnostics with explicitly defined denominators."""
    total = len(records)
    accepted = sum(bool(r["accepted"]) for r in records)
    initially_feasible = sum(bool(r["initially_feasible"]) for r in records)
    conditional_accepted = sum(bool(r["accepted"] and r["initially_feasible"]) for r in records)
    reached = sum(bool(r["reached_destination"]) for r in records)
    embedding_success = sum(bool(r["accepted"] and r["reached_destination"]) for r in records)
    categories = Counter(r["failure_category"] for r in records)
    rewards = [r["total_reward"] for r in records]
    entropies = [r["entropy"] for r in records if r["policy_steps"]]
    return {
        "requests": total, "accepted_count": accepted, "rejected_count": total - accepted,
        "overall_success_rate": _ratio(accepted, total),
        "initially_feasible_count": initially_feasible,
        "initially_feasible_rate": _ratio(initially_feasible, total),
        "conditional_success_rate": _ratio(conditional_accepted, initially_feasible),
        "reached_destination_count": reached,
        "vnf_embedding_success_rate": _ratio(embedding_success, reached),
        "no_action_termination_rate": _ratio(sum(r["no_action_termination"] for r in records), total),
        "average_policy_steps": mean(r["policy_steps"] for r in records) if records else None,
        "policy_entropy_mean": mean(entropies) if entropies else None,
        "policy_entropy_std": pstdev(entropies) if entropies else None,
        "reward_mean": mean(rewards) if rewards else None,
        "reward_std": pstdev(rewards) if rewards else None,
        "initial_endpoint_cpu_infeasible_rate": _ratio(categories["initial_endpoint_cpu_infeasible"], total),
        "full_vnf_placement_failure_rate": _ratio(categories["vnf_placement_failure"], total),
        "failure_categories": dict(categories),
        "failure_category_rates": {k: _ratio(v, total) for k, v in categories.items()},
        "denominators": {"overall": total, "initially_feasible": initially_feasible,
                         "vnf_embedding": reached, "failure_rates": total},
    }


def validation_pass(config, requests, agent, seed):
    """Deterministic held-out validation; creates no gradients and performs no update."""
    scenarios = {
        "frozen_no_failure": FrozenTopology(LEOConstellation(config.constellation).snapshot(0)),
        "orbital": LEOConstellation(config.constellation),
        "random_failure": FaultTopology(LEOConstellation(config.constellation), probability=.1,
                                        slot_s=config.topology_step_s, seed=seed),
    }
    results = {}
    for name, topology in scenarios.items():
        env = SFCOrchestrationEnv(topology, config.mdp,
            cpu_load_weight=config.cpu_load_weight, bandwidth_load_weight=config.bandwidth_load_weight)
        collector = RolloutCollector(env, agent, seed=seed, reward_scale=config.training.reward_scale)
        rollouts = []
        started = perf_counter()
        for request in requests:
            advance_training_time(env, request.arrival_s, config.topology_step_s)
            rollouts.append(collector.collect(request, deterministic=True, track_grad=False))
        losses = batch_request_losses(rollouts, config.training.entropy_coefficient,
                                      config.training.advantage_epsilon)
        records = []
        for rollout in rollouts:
            final = rollout.transitions[-1]
            initial = final.info["initial_diagnostics"]
            records.append({"accepted": rollout.accepted, "initially_feasible": initial["route_feasible"] and initial["endpoint_cpu_feasible"],
                "reached_destination": final.next_state["path"][-1] == final.state["request"]["destination"],
                "no_action_termination": final.action is None, "failure_category": final.info["failure_category"],
                "total_reward": rollout.total_reward, "policy_steps": sum(t.action is not None for t in rollout.transitions),
                "entropy": mean(float(t.entropy) for t in rollout.transitions if t.entropy is not None) if any(t.entropy is not None for t in rollout.transitions) else 0.0})
        summary = summarize_records(records)
        neighbors = sum(int(t.graph.action_mask.sum()) for r in rollouts for t in r.transitions)
        summary.update(actor_loss=float(losses.actor), critic_loss=float(losses.critic),
            mean_abs_td_error=mean(float(t.advantage.abs()) for r in rollouts for t in r.transitions),
            feasible_action_ratio=_ratio(sum(int(t.graph.feasible_action_mask.sum()) for r in rollouts for t in r.transitions), neighbors),
            jain_fairness=env.resources.fairness(), seconds_per_100_sfc=(perf_counter()-started)*100/len(requests))
        results[name] = summary
    return results


def evaluate(config, output_dir, *, epochs=20, requests_per_epoch=100, seed=7, plots=False,
             training_mode="epoch_reset"):
    if type(epochs) is not int or epochs <= 0:
        raise ValueError("epochs must be positive")
    if training_mode not in ("epoch_reset", "persistent"):
        raise ValueError("Unknown training mode")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    # CPU 單執行緒使小圖推論避免執行緒開销；主程式負責設定，不改函式呼叫者設定。
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        agent = ActorCriticAgent(config.constellation.satellites, config.gat_features, config.gat, config.agent)
    env = SFCOrchestrationEnv(LEOConstellation(config.constellation), config.mdp,
                             cpu_load_weight=config.cpu_load_weight, bandwidth_load_weight=config.bandwidth_load_weight)
    trainer = A2CTrainer(env, agent, config.training, seed=seed + 100)
    # 固定 held-out 測試種子；不用測試集選模型或調整訓練參數。
    testing, eval_config = workload(config, 100, seed + 100000)
    initial = run_dynamic(eval_config, testing, DeploymentPolicy(agent.actor, config.mdp), repair=False)
    history, per_request, validation_history = [], [], []
    start = perf_counter()
    for epoch in range(1, epochs + 1):
        if training_mode == "epoch_reset":
            env.reset()
        batch = training_batch(config, requests_per_epoch, seed + 1000 + epoch, epoch,
                               training_mode, env.resources.time_s)
        epoch_start = perf_counter()
        records = []
        size = config.training.update_requests_per_batch
        for start_index in range(0, len(batch), size):
            pending = []
            batch_started = perf_counter()
            for q in batch[start_index:start_index + size]:
                advance_training_time(env, q.arrival_s, config.topology_step_s)
                pending.append(trainer.collector.collect(q, track_grad=True))
            records.extend(trainer.train_rollouts(pending, elapsed_seconds=perf_counter()-batch_started))
        epoch_seconds = perf_counter() - epoch_start
        rewards = [r["total_reward"] for r in records]
        neighbors = sum(r["neighbor_action_count"] for r in records)
        row = {"epoch": epoch, "requests": len(batch),
               "cumulative_reward": sum(r["total_reward"] for r in records),
               "mean_reward": mean(r["total_reward"] for r in records),
               "success_rate": mean(r["accepted"] for r in records),
               "mean_actor_loss": mean(r["actor_loss"] for r in records),
               "mean_critic_loss": mean(r["critic_loss"] for r in records),
               "reward_std": pstdev(rewards), "reward_min": min(rewards), "reward_max": max(rewards),
               "accepted_count": sum(r["accepted"] for r in records),
               "rejected_count": sum(not r["accepted"] for r in records),
               "invalid_action_count": sum(r["invalid_action_count"] for r in records),
               "feasible_action_ratio": sum(r["feasible_action_count"] for r in records) / neighbors if neighbors else 0.0,
               "training_seconds": epoch_seconds,
               "training_seconds_per_100_sfc": epoch_seconds * 100 / len(records)}
        for key in ("mean_advantage", "std_advantage", "mean_td_error", "mean_abs_td_error",
                    "reward_before_normalization", "reward_after_normalization"):
            row[key] = mean(r[key] for r in records)
        categories = Counter(r["failure_category"] for r in records)
        for key in ("initial_no_physical_route", "initial_delay_infeasible", "initial_bandwidth_infeasible",
                    "initial_bandwidth_delay_infeasible", "initial_endpoint_cpu_infeasible",
                    "policy_delay_failure", "vnf_placement_failure", "policy_route_or_placement_failure"):
            row[key] = categories[key]
        row["initial_route_feasible_count"] = sum(r["initial_route_feasible"] for r in records)
        row["delay_failure_rate"] = sum(r["terminal_info"]["reason"] == "delay_exceeded" for r in records) / len(records)
        row.update({k: v for k, v in summarize_records(records).items()
                    if k not in ("requests", "accepted_count", "rejected_count", "reward_std")})
        history.append(row)
        per_request.extend({"epoch": epoch, **{k: r[k] for k in
                            ("request_id", "accepted", "total_reward", "actor_loss", "critic_loss",
                             "steps", "policy_steps", "mean_advantage", "std_advantage", "mean_td_error",
                             "mean_abs_td_error", "reward_before_normalization", "reward_after_normalization",
                             "feasible_action_ratio", "invalid_action_count", "training_seconds",
                             "failure_category", "initial_route_feasible", "initially_feasible",
                             "reached_destination", "no_action_termination", "entropy")},
                             "reason": r["terminal_info"]["reason"]} for r in records)
        if epoch % 5 == 0 or epoch == epochs:
            validation_history.append({"epoch": epoch, "scenarios": validation_pass(
                eval_config, testing, agent, seed + 300000)})
        print(json.dumps(row), flush=True)
    training_seconds = perf_counter() - start
    checkpoint = output / "policy.pt"
    save_policy(checkpoint, agent, config.mdp)
    # 正式部署評估從磁碟重載，驗證 checkpoint → inference 的完整路徑。
    scenarios = {}
    migration_config = MigrationConfig()
    for name, probability in (("orbital", 0.0), ("random_failure", .1)):
        topology = FaultTopology(LEOConstellation(config.constellation), probability=probability,
                                 slot_s=config.topology_step_s, seed=seed + 200000)
        for policy_name in ("gahso", "shortest_delay"):
            for repair in (False, True):
                policy = (load_policy(checkpoint, expected_num_nodes=config.constellation.satellites)
                          if policy_name == "gahso" else partial(shortest_delay_greedy,
                          endpoint_placement_mode=config.mdp.endpoint_placement_mode))
                key = f"{name}_{policy_name}_{'repair' if repair else 'no_repair'}"
                scenarios[key] = run_dynamic(eval_config, testing, policy, topology=topology,
                                            repair=repair, migration_config=migration_config)
                print(json.dumps({"scenario": key, "accepted": scenarios[key]["accepted"],
                                  "migrations": scenarios[key]["vnf_migrations"]}), flush=True)
    report = {
        "config": asdict(config), "evaluation_config": asdict(eval_config),
        "experiment": {"seed": seed, "epochs": epochs, "requests_per_epoch": requests_per_epoch,
                       "test_requests": len(testing), "test_seed": seed + 100000,
                       "fault_seed": seed + 200000, "fault_probability": .1,
                       "migration_config": asdict(migration_config), "device": "cpu",
                       "torch_threads": torch.get_num_threads(), "torch_version": str(torch.__version__),
                       "python": platform.python_version(), "platform": platform.platform(),
                       "training_seconds": training_seconds, "training_mode": training_mode,
                       "optimizer": "Adam", "reward_scale": config.training.reward_scale,
                       "loss_reduction": "mean_per_update_batch",
                       "advantage_normalization": "across_complete_requests_in_update_batch",
                       "update_requests_per_batch": config.training.update_requests_per_batch,
                       "deployment": "greedy_actor_feasible_mask"},
        "training_history": history, "validation_history": validation_history,
        "metric_denominators": {"overall_success_rate": "all requests",
            "initially_feasible_rate": "all requests",
            "conditional_success_rate": "requests initially feasible in route, bandwidth/delay and endpoint CPU",
            "vnf_embedding_success_rate": "requests whose Actor path reached destination",
            "failure_category_rates": "all requests"},
        "convergence": assess_convergence(history),
        "validation_convergence": assess_validation_convergence(validation_history),
        "training_failure_categories": dict(Counter(r["failure_category"] for r in per_request)),
        "topology_audit": audit_topology(LEOConstellation(config.constellation)),
        "initial_held_out": initial, "scenarios": scenarios,
        "limitations": ["Single training seed; no confidence intervals or convergence proof.",
                       "Atomic migration: excludes state-transfer bytes, transfer delay and double CPU.",
                       "Sequential bounded P2 search, not a joint/global or GPU exhaustive optimum.",
                       "Tick-sampled topology; failures last one fault slot; endpoints remain fixed.",
                       "Runtime includes CPU policy planning/embedding/admission; excludes training, loading and repair."],
    }
    (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    for filename, rows in (("training.csv", history), ("training_requests.csv", per_request)):
        with (output / filename).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=rows[0])
            writer.writeheader()
            writer.writerows(rows)
    compact = [{ "scenario": name, **{k: r[k] for k in (
        "acceptance_rate", "mean_arrival_jain", "seconds_per_100_sfc",
        "event_continuity_before_repair", "event_continuity_after_repair",
        "service_time_availability", "uninterrupted_after_repair_rate",
        "repair_attempts", "repair_successes", "vnf_migrations", "reroutes", "limited_searches",
        "accepted", "affected_services", "repair_success_rate", "vnf_migrations_per_accepted_sfc",
        "vnf_migrations_per_affected_sfc")}}
        for name, r in scenarios.items()]
    with (output / "comparison.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=compact[0])
        writer.writeheader()
        writer.writerows(compact)
    if plots:
        plot_history(history, output)
    return report


def plot_history(history, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)
    for ax, key, label in zip(axes.flat,
            ("cumulative_reward", "success_rate", "mean_actor_loss", "mean_critic_loss"),
            ("Cumulative reward per epoch", "Orchestration success rate", "Actor loss (mean/request)", "Critic loss (mean/request)")):
        ax.plot([r["epoch"] for r in history], [r[key] for r in history], marker=".", linewidth=1)
        ax.set(xlabel="Epoch", ylabel=label)
        ax.grid(alpha=.25)
    fig.suptitle("GAHSO training diagnostics (single seed)")
    fig.savefig(Path(output) / "training_curves.png", dpi=180)
    fig.savefig(Path(output) / "training_curves.svg")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("config/default.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/final_25"))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--requests-per-epoch", type=int, default=100)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--plots", action="store_true")
    parser.add_argument("--training-mode", choices=("epoch_reset", "persistent"), default="epoch_reset")
    parser.add_argument("--disable-delay-mask", action="store_true", help="Controlled ablation; same features and traffic")
    parser.add_argument("--endpoint-placement-mode", choices=("fixed", "neighbor_allowed"))
    parser.add_argument("--update-requests-per-batch", type=int)
    args = parser.parse_args()
    if args.threads <= 0:
        parser.error("--threads must be positive")
    torch.set_num_threads(args.threads)
    config = SimulationConfig.from_dict(json.loads(args.config.read_text(encoding="utf-8")))
    if args.disable_delay_mask:
        config = replace(config, mdp=replace(config.mdp, delay_feasibility_mask=False))
    if args.endpoint_placement_mode:
        config = replace(config, mdp=replace(config.mdp,
            endpoint_placement_mode=args.endpoint_placement_mode))
    if args.update_requests_per_batch is not None:
        if args.update_requests_per_batch <= 0:
            parser.error("--update-requests-per-batch must be positive")
        config = replace(config, training=replace(config.training,
            update_requests_per_batch=args.update_requests_per_batch))
    report = evaluate(config, args.output_dir, epochs=args.epochs,
                      requests_per_epoch=args.requests_per_epoch, seed=args.seed, plots=args.plots,
                      training_mode=args.training_mode)
    training_seconds = report["experiment"]["training_seconds"]
    print(json.dumps({"output": str(args.output_dir),
                      "training_seconds": training_seconds,
                      "training_minutes": training_seconds / 60,
                      "convergence": report["convergence"]}, indent=2))


if __name__ == "__main__":
    main()
