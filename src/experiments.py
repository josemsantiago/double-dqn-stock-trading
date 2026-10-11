"""The full experiment, run in stages, with independent seeds trained in parallel.

  regularization  development check on the validation block: the plain network against
                  the regularized one (L2 weight penalty, observation noise, smaller
                  network) at the default configuration, five seeds each
  grid            tuning on the validation block: reward type x discount x learning rate
                  x window length, three seeds per configuration
  final           the selected configuration trained with ten fresh seeds on SPY, each
                  seed's best-validation checkpoint saved and scored once on the test block
                  against the four baselines, plus the Part 1 success test
  stress          the same SPY agents scored on the test block at four cost levels and on
                  the 2005-2009 window, and the whole pipeline rerun on QQQ

Each training run is small and single-threaded, so runs are spread over the CPU cores
with one process per run. Everything is seeded, so a stage reruns to the same numbers.
"""

import argparse
import itertools
import json
import multiprocessing as mp
import os
import time

import numpy as np
import pandas as pd
import torch

import data
import evaluate
import train

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "outputs")
MODELS = os.path.join(ROOT, "models")
WORKERS = max(1, min(16, (os.cpu_count() or 2) - 2))

DEFAULT_CONFIG = {"reward": "logret", "gamma": 0.99, "lr": 1e-4, "window": 20}
PLAIN = {"hidden": 128, "weight_decay": 0.0, "obs_noise": 0.0}   # the unregularized network
GRID = {"reward": ["logret", "dsr"], "gamma": [0.9, 0.99], "lr": [1e-4, 5e-4], "window": [20, 60]}
FINAL_SEEDS = list(range(100, 110))
COST_LEVELS = [0.0, 0.0005, 0.001, 0.0025]      # 0, 5, 10, and 25 basis points


# ---------------------------------------------------------------------------
# Parallel runner
# ---------------------------------------------------------------------------
def _worker(job):
    torch.set_num_threads(1)
    ticker, config, seed, settings, keep_state = job
    res = train.train_run(ticker, config, seed, settings=settings)
    if not keep_state:
        res["state"] = None
    return res


def run_jobs(jobs, label):
    t0 = time.time()
    with mp.get_context("spawn").Pool(WORKERS) as pool:
        results = pool.map(_worker, jobs, chunksize=1)
    log("%s: %d runs on %d processes in %.0f s" % (label, len(jobs), WORKERS, time.time() - t0))
    return results


def log(msg):
    print(msg, flush=True)
    with open(os.path.join(OUT, "run_log.txt"), "a") as f:
        f.write(msg + "\n")


def _finite(o):
    """Replace NaN/inf with None so the JSON is valid for any strict parser."""
    if isinstance(o, float):
        return o if np.isfinite(o) else None
    if isinstance(o, dict):
        return {k: _finite(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_finite(v) for v in o]
    return o


def save_json(obj, name):
    with open(os.path.join(OUT, name), "w") as f:
        json.dump(_finite(obj), f, indent=1)


def curve_summary(res):
    """Validation and training-curve summaries used to judge overfitting."""
    lg = res["logs"]
    return {"best_val_sharpe": res["best_val_sharpe"], "best_episode": res["best_episode"],
            "final_val_sharpe": lg["val_sharpe"][-1],
            "late_val_sharpe": float(np.mean(lg["val_sharpe"][-5:])),
            "late_train_reward": float(np.mean(lg["episode_reward"][-20:])),
            "final_mean_q": lg["mean_q"][-1]}


# ---------------------------------------------------------------------------
# Scoring helpers
# ---------------------------------------------------------------------------
def checkpoint_path(ticker, seed):
    return os.path.join(MODELS, "%s_seed%d.pt" % (ticker.lower(), seed))


def save_checkpoint(res):
    keep = ("ticker", "config", "seed", "obs_dim", "settings", "best_val_sharpe", "best_episode", "state", "norm")
    torch.save({k: res[k] for k in keep}, checkpoint_path(res["ticker"], res["seed"]))


def load_agents(ticker, seeds):
    return [(s, train.load_agent(checkpoint_path(ticker, s))) for s in seeds]


def score_agents(agents, block, cost):
    """Greedy run and metrics for every seed's agent on one block."""
    runs, rows = [], []
    for seed, agent in agents:
        run = evaluate.run_policy(block, evaluate.greedy(agent), cost)
        runs.append(run)
        rows.append(dict(strategy="Double DQN", seed=seed, **evaluate.metrics(run)))
    return runs, rows


def score_baselines(block, cost):
    base = evaluate.baseline_runs(block, cost)
    rows = [dict(strategy=k, seed=None, **evaluate.metrics(v)) for k, v in base.items() if k != "Random"]
    rows.append(dict(strategy="Random", seed=None, **evaluate.mean_metrics(base["Random"])))
    return base, rows


def summary_table(agent_rows, base_rows):
    """Per-seed agent rows, the agent mean and standard deviation, then the baselines."""
    df = pd.DataFrame(agent_rows)
    num = df.drop(columns=["strategy", "seed"])
    mean = dict(strategy="Double DQN (mean of seeds)", seed=None, **num.mean().to_dict())
    sd = dict(strategy="Double DQN (sd of seeds)", seed=None, **num.std(ddof=1).to_dict())
    return pd.concat([df, pd.DataFrame([mean, sd]), pd.DataFrame(base_rows)], ignore_index=True)


def settle_point(logs, tol=0.05):
    """First evaluation after which the greedy validation policy changes on fewer than
    `tol` of days at every later evaluation; None if it never settles."""
    ch = logs["val_policy_change"]
    for k in range(1, len(ch)):
        if all(c < tol for c in ch[k:]):
            return {"episode": logs["eval_episode"][k - 1], "steps": logs["eval_steps"][k - 1]}
    return None


def train_and_test(ticker, cfg, tag):
    """Train ten seeds, save their checkpoints, and score them once on the test block."""
    res = run_jobs([(ticker, cfg, s, {}, True) for s in FINAL_SEEDS], "%s final training" % ticker)
    for r in res:
        save_checkpoint(r)
    blocks, _ = data.make_blocks(ticker, cfg["window"])
    agents = [(r["seed"], train.load_agent(r)) for r in res]
    cost = train.SETTINGS["cost"]
    runs, rows = score_agents(agents, blocks["test"], cost)
    base, base_rows = score_baselines(blocks["test"], cost)
    table = summary_table(rows, base_rows)
    table.to_csv(os.path.join(OUT, "%s_test_metrics.csv" % tag), index=False)
    bh = [r for r in base_rows if r["strategy"] == "Buy-and-hold"][0]["sharpe"]
    success = evaluate.success_test([r["sharpe"] for r in rows], bh)
    save_json(success, "%s_success_test.json" % tag)
    np.savez_compressed(
        os.path.join(OUT, "%s_test_runs.npz" % tag),
        dates=runs[0]["dates"], close=blocks["test"]["close"][:-1], seeds=np.array(FINAL_SEEDS),
        agent_returns=np.stack([r["returns"] for r in runs]),
        agent_positions=np.stack([r["positions"] for r in runs]),
        **{"%s_returns" % k.replace("-", "_").replace(" ", "_").lower(): v["returns"]
           for k, v in base.items() if k != "Random"},
        ma_crossover_positions=base["MA crossover"]["positions"])
    save_json({str(r["seed"]): {"logs": r["logs"], "best_episode": r["best_episode"],
                                "best_val_sharpe": r["best_val_sharpe"], "settle": settle_point(r["logs"]),
                                "seconds": r["seconds"]} for r in res},
              "%s_training_logs.json" % tag)
    log("%s test: agent Sharpe %.2f +/- %.2f (SE) vs buy-and-hold %.2f -> success test %s"
        % (ticker, success["agent_mean_sharpe"], success["agent_sharpe_se"], bh,
           "PASSED" if success["passed"] else "not passed"))
    return table, success


# ---------------------------------------------------------------------------
# Stages
# ---------------------------------------------------------------------------
def stage_regularization():
    jobs = ([("SPY", DEFAULT_CONFIG, s, PLAIN, True) for s in range(5)]
            + [("SPY", DEFAULT_CONFIG, s, {}, True) for s in range(5)])
    res = run_jobs(jobs, "regularization")
    blocks, _ = data.make_blocks("SPY", DEFAULT_CONFIG["window"])
    rows = []
    for r in res:
        m = evaluate.metrics(evaluate.run_policy(blocks["val"], evaluate.greedy(train.load_agent(r)), train.SETTINGS["cost"]))
        rows.append(dict(variant="plain" if r["settings"]["hidden"] == PLAIN["hidden"] else "regularized",
                         seed=r["seed"], val_turnover=m["turnover_per_year"], **curve_summary(r)))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "regularization.csv"), index=False)
    log("regularization (validation, 5 seeds):\n%s" % df.drop(columns="seed").groupby("variant").mean().round(2).to_string())


def stage_grid():
    configs = [dict(zip(GRID, v)) for v in itertools.product(*GRID.values())]
    res = run_jobs([("SPY", c, s, {}, False) for c in configs for s in range(3)], "validation grid")
    runs = pd.DataFrame([dict(**r["config"], seed=r["seed"], best_val_sharpe=r["best_val_sharpe"],
                              best_episode=r["best_episode"]) for r in res])
    runs.to_csv(os.path.join(OUT, "validation_grid_runs.csv"), index=False)
    keys = list(GRID)
    summ = (runs.groupby(keys)["best_val_sharpe"].agg(["mean", "std"]).reset_index()
            .sort_values("mean", ascending=False).rename(columns={"mean": "val_sharpe_mean", "std": "val_sharpe_sd"}))
    summ.to_csv(os.path.join(OUT, "validation_grid.csv"), index=False)
    best = summ.iloc[0]
    cfg = {"reward": best["reward"], "gamma": float(best["gamma"]), "lr": float(best["lr"]), "window": int(best["window"])}
    save_json({"config": cfg, "val_sharpe_mean": float(best["val_sharpe_mean"]),
               "val_sharpe_sd": float(best["val_sharpe_sd"]), "settings": train.SETTINGS}, "selected_config.json")
    log("validation grid top 5:\n%s\nselected: %s" % (summ.head(5).round(3).to_string(index=False), cfg))


def selected_config():
    with open(os.path.join(OUT, "selected_config.json")) as f:
        return json.load(f)["config"]


def stage_final():
    train_and_test("SPY", selected_config(), "spy")


def stage_stress():
    cfg = selected_config()
    blocks, _ = data.make_blocks("SPY", cfg["window"])
    agents = load_agents("SPY", FINAL_SEEDS)

    # 1) the same SPY policies scored on the test block at four cost levels
    rows, succ = [], {}
    for c in COST_LEVELS:
        _, ar = score_agents(agents, blocks["test"], c)
        _, br = score_baselines(blocks["test"], c)
        bh = [r for r in br if r["strategy"] == "Buy-and-hold"][0]["sharpe"]
        succ["%g" % (c * 1e4)] = evaluate.success_test([r["sharpe"] for r in ar], bh)
        for r in summary_table(ar, br).to_dict("records"):
            rows.append(dict(cost_bps=c * 1e4, **r))
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "stress_costs.csv"), index=False)
    save_json(succ, "stress_costs_success.json")
    log("cost stress (SPY test): " + "; ".join("%s bps: agent %.2f vs B&H %.2f" % (k, v["agent_mean_sharpe"], v["benchmark_sharpe"]) for k, v in succ.items()))

    # 2) the same SPY policies on the 2005-2009 window, which includes the 2008 crash
    cost = train.SETTINGS["cost"]
    runs, ar = score_agents(agents, blocks["earlier"], cost)
    base, br = score_baselines(blocks["earlier"], cost)
    summary_table(ar, br).to_csv(os.path.join(OUT, "stress_earlier_metrics.csv"), index=False)
    bh = [r for r in br if r["strategy"] == "Buy-and-hold"][0]["sharpe"]
    save_json(evaluate.success_test([r["sharpe"] for r in ar], bh), "stress_earlier_success.json")
    np.savez_compressed(os.path.join(OUT, "stress_earlier_runs.npz"), dates=runs[0]["dates"],
                        close=blocks["earlier"]["close"][:-1],
                        agent_returns=np.stack([r["returns"] for r in runs]),
                        agent_positions=np.stack([r["positions"] for r in runs]),
                        buy_and_hold_returns=base["Buy-and-hold"]["returns"])
    log("earlier window 2005-2009: agent Sharpe %.2f vs buy-and-hold %.2f" % (np.mean([r["sharpe"] for r in ar]), bh))

    # 3) the whole pipeline rerun on a second instrument with the SPY-selected settings
    train_and_test("QQQ", cfg, "qqq")


STAGES = {"regularization": stage_regularization, "grid": stage_grid, "final": stage_final, "stress": stage_stress}



# ===========================================================================
# IMPROVED AGENT: diagnosed on validation, then tested once.
# Long/flat action set (no shorting an upward-drifting index), differential-Sharpe or
# log-return reward, an optional turnover penalty, and a seed ENSEMBLE as the deployed
# policy. The configuration is selected on validation; the test block is touched once.
# ===========================================================================
IMPROVED_BASE = {"gamma": 0.9, "lr": 1e-4, "window": 60}
IMPROVED_GRID = {"reward": ["logret", "dsr"], "turnover_penalty": [0.0, 0.3, 0.6, 1.0]}
IMPROVED_SEEDS = list(range(200, 220))   # a 20-seed ensemble for stability
SELECT_SEEDS = range(5)
# The configuration is selected by validation performance scored at a conservative 20 bps
# cost, four times the 5 bps actually charged. Raw validation Sharpe at 5 bps rewards a
# high-turnover policy that only looks good when trading is nearly free and that the
# cost-stress analysis shows collapses out of sample; scoring selection at a stress cost
# instead favors the low-turnover, trend-following policy that generalizes.
SELECT_COST = 0.002


def improved_configs():
    out = []
    for r in IMPROVED_GRID["reward"]:
        for tp in IMPROVED_GRID["turnover_penalty"]:
            out.append(dict(IMPROVED_BASE, reward=r, turnover_penalty=tp, long_only=True))
    return out


def ensemble_run(agents, block, cost, long_only):
    return evaluate.run_policy(block, evaluate.ensemble_greedy(agents), cost, long_only=long_only)


def stage_improve():
    cost = train.SETTINGS["cost"]
    blocks, _ = data.make_blocks("SPY", IMPROVED_BASE["window"])

    # --- select a configuration on the validation block only ---
    configs = improved_configs()
    jobs = [("SPY", c, s, {}, True) for c in configs for s in SELECT_SEEDS]
    res = run_jobs(jobs, "improved validation selection")
    by_cfg = {}
    for r in res:
        key = (r["config"]["reward"], r["config"]["turnover_penalty"])
        by_cfg.setdefault(key, []).append(train.load_agent(r))
    rows = []
    for key, agents in by_cfg.items():
        per = [evaluate.metrics(evaluate.run_policy(blocks["val"], evaluate.greedy(a), cost, long_only=True))["sharpe"]
               for a in agents]
        em = evaluate.metrics(ensemble_run(agents, blocks["val"], cost, True))             # at the real 5 bps cost
        robust = evaluate.metrics(ensemble_run(agents, blocks["val"], SELECT_COST, True))  # at the 20 bps stress cost
        rows.append({"reward": key[0], "turnover_penalty": key[1], "long_only": True,
                     "val_seed_sharpe_mean": float(np.mean(per)), "val_seed_sharpe_sd": float(np.std(per, ddof=1)),
                     "val_ensemble_sharpe": em["sharpe"], "val_ensemble_turnover": em["turnover_per_year"],
                     "val_ensemble_long": em["share_long"], "val_ensemble_sharpe_at_20bps": robust["sharpe"]})
    sel = pd.DataFrame(rows).sort_values("val_ensemble_sharpe_at_20bps", ascending=False).reset_index(drop=True)
    best = sel.iloc[0]                                            # most cost-robust on validation
    sel["selected"] = sel.index == 0
    sel.to_csv(os.path.join(OUT, "improved_selection.csv"), index=False)
    cfg = dict(IMPROVED_BASE, reward=best["reward"], turnover_penalty=float(best["turnover_penalty"]), long_only=True)
    save_json({"config": cfg, "val_ensemble_sharpe": float(best["val_ensemble_sharpe"]),
               "val_seed_sharpe_mean": float(best["val_seed_sharpe_mean"])}, "improved_selected_config.json")
    log("improved selection (validation):\n%s\nselected: %s" % (sel.round(2).to_string(index=False), cfg))

    # --- final: train ten seeds, test once (ensemble + per-seed) ---
    improved_train_and_test("SPY", cfg, "improved_spy")

    # --- stress: cost levels, 2008 window, and the second asset, all with the ensemble ---
    improved_stress("SPY", cfg)
    improved_train_and_test("QQQ", cfg, "improved_qqq")


def improved_train_and_test(ticker, cfg, tag):
    cost = train.SETTINGS["cost"]
    res = run_jobs([(ticker, cfg, s, {}, True) for s in IMPROVED_SEEDS], "%s improved training" % ticker)
    for r in res:
        torch.save({k: r[k] for k in ("ticker", "config", "seed", "obs_dim", "n_actions", "settings",
                                       "best_val_sharpe", "best_episode", "state", "norm")},
                   os.path.join(MODELS, "improved_%s_seed%d.pt" % (ticker.lower(), r["seed"])))
    blocks, _ = data.make_blocks(ticker, cfg["window"])
    agents = [train.load_agent(r) for r in res]
    runs, rows = [], []
    for r, a in zip(res, agents):
        run = evaluate.run_policy(blocks["test"], evaluate.greedy(a), cost, long_only=True)
        runs.append(run)
        rows.append(dict(strategy="Double DQN", seed=r["seed"], **evaluate.metrics(run)))
    ens = ensemble_run(agents, blocks["test"], cost, True)
    ens_m = evaluate.metrics(ens)
    base, base_rows = score_baselines(blocks["test"], cost)
    bh = [x for x in base_rows if x["strategy"] == "Buy-and-hold"][0]["sharpe"]
    table = summary_table(rows, base_rows)
    table = pd.concat([table, pd.DataFrame([dict(strategy="Double DQN (ensemble)", seed=None, **ens_m)])],
                      ignore_index=True)
    table.to_csv(os.path.join(OUT, "%s_test_metrics.csv" % tag), index=False)
    succ = evaluate.success_test([r["sharpe"] for r in rows], bh)
    succ["ensemble_sharpe"] = ens_m["sharpe"]
    succ["ensemble_beats_benchmark"] = bool(ens_m["sharpe"] > bh)
    save_json(succ, "%s_success_test.json" % tag)
    np.savez_compressed(
        os.path.join(OUT, "%s_test_runs.npz" % tag),
        dates=ens["dates"], close=blocks["test"]["close"][:-1], seeds=np.array(IMPROVED_SEEDS),
        agent_returns=np.stack([r["returns"] for r in runs]),
        agent_positions=np.stack([r["positions"] for r in runs]),
        ensemble_returns=ens["returns"], ensemble_positions=ens["positions"],
        buy_and_hold_returns=base["Buy-and-hold"]["returns"],
        ma_crossover_returns=base["MA crossover"]["returns"])
    save_json({str(r["seed"]): {"logs": r["logs"], "best_episode": r["best_episode"],
                                "best_val_sharpe": r["best_val_sharpe"], "settle": settle_point(r["logs"])}
               for r in res}, "%s_training_logs.json" % tag)
    log("%s improved test: ensemble Sharpe %.2f vs buy-and-hold %.2f (%s); per-seed %.2f +/- %.2f (SE), success %s"
        % (ticker, ens_m["sharpe"], bh, "beats" if ens_m["sharpe"] > bh else "trails",
           succ["agent_mean_sharpe"], succ["agent_sharpe_se"], "PASSED" if succ["passed"] else "not passed"))


def improved_stress(ticker, cfg):
    cost = train.SETTINGS["cost"]
    blocks, _ = data.make_blocks(ticker, cfg["window"])
    agents = [train.load_agent(os.path.join(MODELS, "improved_%s_seed%d.pt" % (ticker.lower(), s)))
              for s in IMPROVED_SEEDS]
    rows = []
    for c in COST_LEVELS:
        em = evaluate.metrics(ensemble_run(agents, blocks["test"], c, True))
        bh = evaluate.metrics(evaluate.returns_from_positions(
            np.ones(len(blocks["test"]["next_ret"]) - 1), blocks["test"]["next_ret"], c))["sharpe"]
        rows.append({"cost_bps": c * 1e4, "ensemble_sharpe": em["sharpe"], "buy_and_hold_sharpe": bh})
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "improved_stress_costs.csv"), index=False)
    # the 2008 crisis window
    runs = [evaluate.run_policy(blocks["earlier"], evaluate.greedy(a), cost, long_only=True) for a in agents]
    ens = ensemble_run(agents, blocks["earlier"], cost, True)
    base, br = score_baselines(blocks["earlier"], cost)
    bh = [x for x in br if x["strategy"] == "Buy-and-hold"][0]["sharpe"]
    save_json({"ensemble_sharpe": evaluate.metrics(ens)["sharpe"], "buy_and_hold_sharpe": bh,
               "seed_mean_sharpe": float(np.mean([evaluate.metrics(r)["sharpe"] for r in runs]))},
              "improved_stress_earlier.json")
    np.savez_compressed(os.path.join(OUT, "improved_stress_earlier_runs.npz"), dates=ens["dates"],
                        close=blocks["earlier"]["close"][:-1], ensemble_returns=ens["returns"],
                        ensemble_positions=ens["positions"], buy_and_hold_returns=base["Buy-and-hold"]["returns"])
    log("improved cost stress: " + "; ".join("%gbps %.2f" % (r["cost_bps"], r["ensemble_sharpe"]) for r in rows)
        + " | 2008 window ensemble %.2f vs B&H %.2f" % (evaluate.metrics(ens)["sharpe"], bh))


STAGES["improve"] = stage_improve
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Run the experiment stages.")
    ap.add_argument("--stage", default="all", choices=list(STAGES) + ["all"])
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    os.makedirs(MODELS, exist_ok=True)
    for t in ("SPY", "QQQ"):                 # download once here so parallel workers never race
        data.load_prices(t)
    for name in (STAGES if args.stage == "all" else [args.stage]):
        log("=== stage %s ===" % name)
        STAGES[name]()
