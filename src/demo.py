"""The recorded demo: the improved agent (long/flat ensemble) trading 2021-2023.

Plays one representative seed's greedy policy forward one day at a time. The top panel
is the SPY price, shaded green on days the agent held long and rust on days it held
short, with a dot marking the current day. The bottom panel grows the cumulative return
of the account next to buy-and-hold. The clip shows what the agent does, day by day, and
that it falls behind a passive position. Writes demo/agent_trading_demo.mp4 and a still
frame demo/demo_still.png for the slide.
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.ticker import FuncFormatter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "outputs")
DEMO = os.path.join(ROOT, "demo")

AGENT, BH, INK, INK_SOFT, GRID = "#0F8A5F", "#2a78d6", "#1F2933", "#6B7280", "#E4E7EA"
LONG, SHORT = "#0F8A5F", "#C2571A"
FPS, STEP = 25, 2        # 2 trading days per frame -> about 15 seconds


def _seed_index(z):
    finals = [np.cumprod(1 + r)[-1] for r in z["agent_returns"]]
    return int(np.argsort(finals)[len(finals) // 2])     # the median seed by final return


def build():
    os.makedirs(DEMO, exist_ok=True)
    z = np.load(os.path.join(OUT, "improved_spy_test_runs.npz"), allow_pickle=False)
    dates, close = z["dates"], z["close"]
    pos = z["ensemble_positions"]
    agent_eq = np.cumprod(1 + z["ensemble_returns"]) - 1.0
    bh_eq = np.cumprod(1 + z["buy_and_hold_returns"]) - 1.0
    n = len(dates)
    years = np.array([d[:4] for d in dates])
    xticks = [np.where(years == y)[0][0] for y in sorted(set(years))]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6.2), gridspec_kw={"height_ratios": [1.35, 1]}, dpi=120)
    fig.subplots_adjust(left=0.085, right=0.97, top=0.9, bottom=0.08, hspace=0.42)
    for ax in (ax1, ax2):
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(True, color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        ax.set_xlim(0, n - 1)
        ax.set_xticks(xticks)
        ax.set_xticklabels(sorted(set(years)))
    ax1.set_ylim(close.min() * 0.98, close.max() * 1.02)
    ax1.set_ylabel("SPY price (USD)")
    lo = min(agent_eq.min(), bh_eq.min()) - 0.03
    hi = max(agent_eq.max(), bh_eq.max()) + 0.03
    ax2.set_ylim(lo, hi)
    ax2.set_ylabel("Cumulative return")
    ax2.yaxis.set_major_formatter(FuncFormatter(lambda v, _: "%+.0f%%" % (v * 100)))
    ax2.axhline(0, color=INK_SOFT, lw=0.8)

    price_line, = ax1.plot([], [], color=INK_SOFT, lw=1.4)
    dot, = ax1.plot([], [], "o", color=INK, ms=7, zorder=6)
    agent_line, = ax2.plot([], [], color=AGENT, lw=2.2, label="Double DQN agent")
    bh_line, = ax2.plot([], [], color=BH, lw=1.8, label="Buy-and-hold")
    ax2.legend(frameon=False, loc="lower left", fontsize=10)
    title = fig.suptitle("", x=0.085, ha="left", fontsize=14, color=INK, weight="bold")
    posbox = ax1.text(0.985, 0.06, "", transform=ax1.transAxes, ha="right", va="bottom",
                      fontsize=12, weight="bold")
    shades = []

    def frame(k):
        t = min(k * STEP, n - 1)
        for s in shades:
            s.remove()
        shades.clear()
        x = np.arange(t + 1)
        price_line.set_data(x, close[:t + 1])
        dot.set_data([t], [close[t]])
        agent_line.set_data(x, agent_eq[:t + 1])
        bh_line.set_data(x, bh_eq[:t + 1])
        p = pos[:t + 1]
        shades.append(ax1.fill_between(x, close.min() * 0.98, close.max() * 1.02, where=(p > 0),
                                       color=LONG, alpha=0.14, linewidth=0))
        shades.append(ax1.fill_between(x, close.min() * 0.98, close.max() * 1.02, where=(p < 0),
                                       color=SHORT, alpha=0.16, linewidth=0))
        state = "LONG" if pos[t] > 0 else ("SHORT" if pos[t] < 0 else "FLAT")
        col = LONG if pos[t] > 0 else (SHORT if pos[t] < 0 else INK_SOFT)
        posbox.set_text("%s  ·  position: %s" % (dates[t], state))
        posbox.set_color(col)
        title.set_text("Double-DQN agent trading SPY, 2021–2023   (agent %+.0f%%  vs  buy-and-hold %+.0f%%)"
                       % (agent_eq[t] * 100, bh_eq[t] * 100))
        return price_line, dot, agent_line, bh_line, posbox, title

    n_frames = n // STEP + 1
    frame(n_frames - 1)
    fig.savefig(os.path.join(DEMO, "demo_still.png"), dpi=120)
    anim = animation.FuncAnimation(fig, frame, frames=n_frames, interval=1000 / FPS, blit=False)
    writer = animation.FFMpegWriter(fps=FPS, bitrate=2400,
                                    metadata={"title": "Double-DQN SPY trading demo"})
    out = os.path.join(DEMO, "agent_trading_demo.mp4")
    anim.save(out, writer=writer, dpi=120)
    plt.close(fig)
    print("wrote %s (%d frames, ~%.0fs) and demo_still.png"
          % (os.path.relpath(out, ROOT), n_frames, n_frames / FPS))


if __name__ == "__main__":
    build()
