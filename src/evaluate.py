"""Rolling a policy through a block, the four baselines, and the trading metrics.

Every strategy is scored on the same net daily returns: the position each day times the
next day's return, less the proportional cost on any change of position, starting flat.
The agent is rolled through the environment so it sees the right observations; the
baselines are position paths scored by the same arithmetic, so they do not depend on the
agent's action set. The metrics always use the real returns, never the shaped training
reward, and cash earns zero.

An ensemble policy takes a majority vote over several seeds' greedy actions, which is
the single most effective way to tame the seed-to-seed variance of a
deep Q-network on noisy financial data.

The success test is the one fixed in the Part 1 plan: the agent's mean annualized Sharpe
ratio across seeds, after costs, has to beat buy-and-hold on the test block by more than
one standard error of the seed spread.
"""

import numpy as np
import torch

from env import TradingEnv


# ---------------------------------------------------------------------------
# Scoring a position path and rolling the agent
# ---------------------------------------------------------------------------
def returns_from_positions(positions, next_ret, cost):
    """Net daily returns of a position path, charged the cost on every change, from flat."""
    pos = np.asarray(positions, dtype=float)
    nr = np.asarray(next_ret[:len(pos)], dtype=float)
    prev = np.concatenate([[0.0], pos[:-1]])
    traded = np.abs(pos - prev)
    return {"returns": pos * nr - cost * traded, "positions": pos,
            "costs": cost * traded, "traded": traded}


def run_policy(block, choose, cost, long_only=False):
    """Step `choose(obs, t)` (an action index) through a whole block greedily, from flat."""
    env = TradingEnv(block, cost=cost, reward="logret", long_only=long_only)
    obs, _ = env.reset(options={"start": 0})
    out = {k: [] for k in ("returns", "positions", "costs", "traded", "dates")}
    done = False
    while not done:
        obs, _, terminated, truncated, info = env.step(choose(obs, env.t))
        for k, v in (("returns", info["net_return"]), ("positions", info["position"]),
                     ("costs", info["cost"]), ("traded", info["traded"]), ("dates", info["date"])):
            out[k].append(v)
        done = terminated or truncated
    return {k: np.array(v) for k, v in out.items()}


def greedy(agent):
    """One agent's greedy policy as a choose(obs, t) function."""
    def choose(obs, t):
        return int(np.argmax(agent.q_values(obs)[0]))
    return choose


def ensemble_greedy(agents):
    """A majority-vote policy over several seeds: each seed votes its greedy action and
    the mode wins. Voting is scale-invariant across seeds and was more robust on the
    validation block than averaging the raw Q-values."""
    def choose(obs, t):
        acts = [int(np.argmax(a.q_values(obs)[0])) for a in agents]
        return int(np.bincount(acts, minlength=agents[0].n_actions).argmax())
    return choose


# ---------------------------------------------------------------------------
# Baselines (position paths, scored by the same arithmetic as the agent)
# ---------------------------------------------------------------------------
def _n(block):
    return len(block["next_ret"]) - 1


def baseline_runs(block, cost, random_seeds=range(10)):
    n = _n(block)
    nr = block["next_ret"]
    runs = {
        "Buy-and-hold": returns_from_positions(np.ones(n), nr, cost),
        "All-cash": returns_from_positions(np.zeros(n), nr, cost),
        "MA crossover": returns_from_positions(
            (block["sma50"][:n] > block["sma200"][:n]).astype(float), nr, cost),
    }
    runs["Random"] = [returns_from_positions(
        np.random.default_rng(10_000 + s).integers(-1, 2, size=n).astype(float), nr, cost)
        for s in random_seeds]
    return runs


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def metrics(run):
    r = run["returns"]
    n = len(r)
    equity = np.concatenate([[1.0], np.cumprod(1.0 + r)])
    sd = r.std(ddof=1)
    sharpe = float(r.mean() / sd * np.sqrt(252.0)) if sd > 1e-12 else 0.0
    cagr = float(equity[-1] ** (252.0 / n) - 1.0)
    drawdown = equity / np.maximum.accumulate(equity) - 1.0
    mdd = float(drawdown.min())
    pos = run["positions"]
    return {
        "cumulative_return": float(equity[-1] - 1.0),
        "annualized_return": cagr,
        "annualized_vol": float(sd * np.sqrt(252.0)),
        "sharpe": sharpe,
        "max_drawdown": mdd,
        "calmar": float(cagr / abs(mdd)) if mdd < -1e-12 else float("nan"),
        "turnover_per_year": float(run["traded"].sum() / n * 252.0),
        "cost_drag_per_year": float(run["costs"].sum() / n * 252.0),
        "profitable_days": float((r > 0).mean()),
        "share_long": float((pos > 0).mean()),
        "share_flat": float((pos == 0).mean()),
        "share_short": float((pos < 0).mean()),
        "days": int(n),
    }


def mean_metrics(runs):
    ms = [metrics(r) for r in runs]
    return {k: float(np.nanmean([m[k] for m in ms])) for k in ms[0]}


def success_test(agent_sharpes, benchmark_sharpe):
    """The Part 1 criterion: mean seed Sharpe must beat the benchmark by more than one SE."""
    s = np.asarray(agent_sharpes, dtype=float)
    mean, se = float(s.mean()), float(s.std(ddof=1) / np.sqrt(len(s)))
    margin = mean - benchmark_sharpe
    return {"agent_mean_sharpe": mean, "agent_sharpe_se": se, "seeds": int(len(s)),
            "benchmark_sharpe": float(benchmark_sharpe), "margin": float(margin),
            "margin_in_se": float(margin / se) if se > 0 else float("nan"),
            "passed": bool(margin > se)}
