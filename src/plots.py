"""The figures, drawn from the saved experiment outputs.

The story is the comparison: the naive agent (short/flat/long, raw reward, one seed)
lost badly, and the improved agent (long/flat, risk-adjusted reward, turnover penalty,
a twenty-seed majority-vote ensemble) recovered to roughly break-even. Both are shown
against buy-and-hold and a moving-average rule. Colors stay fixed: the improved agent is
green, the naive agent muted red, buy-and-hold blue, the moving-average rule orange, cash
or random gray; a position is green when long and rust when short.
"""

import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "outputs")
FIG = os.path.join(ROOT, "figures")

IMPROVED, NAIVE, BH, MA, GRAY = "#0F8A5F", "#C2571A", "#2a78d6", "#eda100", "#8A939B"
LONG, SHORT = "#0F8A5F", "#C2571A"
INK, INK_SOFT, GRID = "#1A1A1A", "#6B7280", "#E4E7EA"

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": INK_SOFT,
    "axes.linewidth": 0.8, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": INK_SOFT, "ytick.color": INK_SOFT, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.8, "figure.dpi": 150, "savefig.dpi": 150,
})


def _style(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_axisbelow(True)
    return ax


def _pct(ax, axis="y"):
    fmt = FuncFormatter(lambda v, _: "%+.0f%%" % (v * 100))
    (ax.yaxis if axis == "y" else ax.xaxis).set_major_formatter(fmt)


def _equity(returns):
    return np.cumprod(1.0 + returns) - 1.0


def _year_ticks(ax, dates):
    years = np.array([d[:4] for d in dates])
    idx = [np.where(years == y)[0][0] for y in sorted(set(years))]
    ax.set_xticks(idx)
    ax.set_xticklabels(sorted(set(years)))


def load_npz(name):
    return np.load(os.path.join(OUT, name), allow_pickle=False)


def _read(csv):
    import csv as _csv
    with open(os.path.join(OUT, csv)) as f:
        return {r["strategy"]: r for r in _csv.DictReader(f)}


# ---------------------------------------------------------------------------
# fig01  test-block equity: naive vs improved vs baselines
# ---------------------------------------------------------------------------
def fig_equity(fname="fig01_test_equity.png"):
    imp = load_npz("improved_spy_test_runs.npz")
    nai = load_npz("spy_test_runs.npz")
    dates = imp["dates"]
    fig, ax = plt.subplots(figsize=(8.4, 4.5))
    _style(ax)
    ax.plot(_equity(nai["agent_returns"].mean(0)), color=NAIVE, lw=1.8, label="Naive DQN (short/flat/long)")
    ax.plot(_equity(imp["ensemble_returns"]), color=IMPROVED, lw=2.2, label="Improved DQN (long/flat ensemble)", zorder=5)
    ax.plot(_equity(imp["buy_and_hold_returns"]), color=BH, lw=1.6, label="Buy-and-hold")
    ax.plot(_equity(imp["ma_crossover_returns"]), color=MA, lw=1.5, label="MA crossover")
    ax.axhline(0.0, color=INK_SOFT, lw=0.8)
    _year_ticks(ax, dates)
    _pct(ax)
    ax.set_ylabel("Cumulative return")
    ax.set_title("From catastrophe to competitive: the improved agent tracks the market",
                 loc="left", fontsize=12.5, color=INK, pad=10)
    ax.legend(frameon=False, loc="lower left", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, fname))
    plt.close(fig)


# ---------------------------------------------------------------------------
# fig02  the recovery: Sharpe and drawdown, naive vs improved vs baselines
# ---------------------------------------------------------------------------
def fig_recovery(fname="fig02_recovery.png"):
    nai = _read("spy_test_metrics.csv")
    imp = _read("improved_spy_test_metrics.csv")
    labels = ["Naive DQN", "Improved DQN", "Buy-and-hold", "MA crossover"]
    colors = [NAIVE, IMPROVED, BH, MA]
    sharpe = [float(nai["Double DQN (mean of seeds)"]["sharpe"]),
              float(imp["Double DQN (mean of seeds)"]["sharpe"]),
              float(imp["Buy-and-hold"]["sharpe"]), float(imp["MA crossover"]["sharpe"])]
    mdd = [float(nai["Double DQN (mean of seeds)"]["max_drawdown"]),
           float(imp["Double DQN (mean of seeds)"]["max_drawdown"]),
           float(imp["Buy-and-hold"]["max_drawdown"]), float(imp["MA crossover"]["max_drawdown"])]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(8.8, 4.2))
    for ax in (a1, a2):
        _style(ax)
    y = np.arange(len(labels))[::-1]
    a1.barh(y, sharpe, color=colors, height=0.6)
    a1.set_yticks(y); a1.set_yticklabels(labels)
    a1.axvline(0, color=INK_SOFT, lw=0.8)
    a1.set_title("Sharpe ratio (after costs)", loc="left", fontsize=11, color=INK)
    for yi, v in zip(y, sharpe):
        a1.text(v + (0.05 if v >= 0 else -0.05), yi, "%+.2f" % v, va="center",
                ha="left" if v >= 0 else "right", fontsize=9.5, color=INK)
    a1.set_xlim(-1.35, 1.35)
    a2.barh(y, mdd, color=colors, height=0.6)
    a2.set_yticks(y); a2.set_yticklabels([])
    a2.set_title("Maximum drawdown", loc="left", fontsize=11, color=INK)
    _pct(a2, "x")
    for yi, v in zip(y, mdd):
        a2.text(v - 0.01, yi, "%.0f%%" % (v * 100), va="center", ha="right", fontsize=9.5, color="white", zorder=6)
    a2.margins(x=0.12)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, fname))
    plt.close(fig)


# ---------------------------------------------------------------------------
# fig03  training diagnostics (improved agent)
# ---------------------------------------------------------------------------
def fig_training(fname="fig03_training.png"):
    with open(os.path.join(OUT, "improved_spy_training_logs.json")) as f:
        logs = json.load(f)

    def mean_curve(key):
        arr = np.array([[np.nan if v is None else v for v in sd["logs"][key]] for sd in logs.values()], dtype=float)
        return np.nanmean(arr, axis=0)

    fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(10.2, 3.5))
    for ax in (a1, a2, a3):
        _style(ax)
    for sd in logs.values():
        lg = sd["logs"]
        a1.plot(range(1, len(lg["episode_reward"]) + 1), lg["episode_reward"], color=IMPROVED, alpha=0.18, lw=0.8)
        a2.plot(range(1, len(lg["mean_q"]) + 1), [np.nan if v is None else v for v in lg["mean_q"]], color=BH, alpha=0.18, lw=0.8)
        a3.plot(lg["eval_episode"], lg["val_sharpe"], color=IMPROVED, alpha=0.22, lw=0.8)
    nL = len(next(iter(logs.values()))["logs"]["episode_reward"])
    a1.plot(range(1, nL + 1), mean_curve("episode_reward"), color=IMPROVED, lw=2.0)
    a2.plot(range(1, nL + 1), mean_curve("mean_q"), color=BH, lw=2.0)
    ev = next(iter(logs.values()))["logs"]["eval_episode"]
    a3.plot(ev, mean_curve("val_sharpe"), color=IMPROVED, lw=2.0)
    a3.axhline(0, color=INK_SOFT, lw=0.8)
    a3.axvspan(0, 100, color=GRAY, alpha=0.12, linewidth=0)
    a1.set_title("Training reward per episode", loc="left", fontsize=10.5, color=INK)
    a2.set_title("Mean predicted Q-value", loc="left", fontsize=10.5, color=INK)
    a3.set_title("Validation Sharpe (gray: not eligible)", loc="left", fontsize=10.5, color=INK)
    for ax in (a1, a2, a3):
        ax.set_xlabel("episode")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, fname))
    plt.close(fig)


# ---------------------------------------------------------------------------
# fig04  learned positions over price (improved ensemble, long/flat)
# ---------------------------------------------------------------------------
def fig_positions(fname="fig04_positions.png"):
    z = load_npz("improved_spy_test_runs.npz")
    dates, close, pos = z["dates"], z["close"], z["ensemble_positions"]
    fig, ax = plt.subplots(figsize=(8.2, 4.0))
    _style(ax)
    x = np.arange(len(dates))
    ax.plot(x, close, color=INK_SOFT, lw=1.3, zorder=3)
    ax.fill_between(x, close.min() * 0.985, close.max() * 1.005, where=(pos > 0),
                    color=LONG, alpha=0.16, linewidth=0)
    _year_ticks(ax, dates)
    ax.set_ylabel("SPY price (USD)")
    ax.set_ylim(close.min() * 0.985, close.max() * 1.005)
    ax.set_title("The improved agent holds long (green) or steps aside, and never shorts",
                 loc="left", fontsize=11.5, color=INK, pad=10)
    handles = [plt.Rectangle((0, 0), 1, 1, color=LONG, alpha=0.5),
               plt.Rectangle((0, 0), 1, 1, color="white", ec=INK_SOFT)]
    ax.legend(handles, ["held long", "flat (in cash)"], frameon=False, loc="upper left", fontsize=9, ncol=2)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, fname))
    plt.close(fig)


# ---------------------------------------------------------------------------
# fig05  cost-level stress (improved ensemble)
# ---------------------------------------------------------------------------
def fig_cost_stress(fname="fig05_cost_stress.png"):
    import csv as _csv
    rows = list(_csv.DictReader(open(os.path.join(OUT, "improved_stress_costs.csv"))))
    costs = [float(r["cost_bps"]) for r in rows]
    agent = [float(r["ensemble_sharpe"]) for r in rows]
    bh = [float(r["buy_and_hold_sharpe"]) for r in rows]
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    _style(ax)
    ax.plot(costs, agent, color=IMPROVED, lw=2.0, marker="o", ms=6, label="Improved DQN")
    ax.plot(costs, bh, color=BH, lw=1.6, marker="s", ms=5, label="Buy-and-hold")
    ax.axhline(0, color=INK_SOFT, lw=0.8)
    ax.set_xlabel("transaction cost (basis points per unit traded)")
    ax.set_ylabel("Sharpe ratio")
    ax.set_title("Low turnover makes the improved agent cost-robust",
                 loc="left", fontsize=12, color=INK, pad=10)
    ax.legend(frameon=False, loc="upper right", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, fname))
    plt.close(fig)


def main():
    os.makedirs(FIG, exist_ok=True)
    fig_equity()
    fig_recovery()
    fig_training()
    fig_positions()
    fig_cost_stress()
    print("figures written to", os.path.relpath(FIG, ROOT))


if __name__ == "__main__":
    main()
