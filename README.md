# Double Deep Q-Network for Single-Asset Stock Trading

A Double Deep Q-Network that learns to trade one liquid asset from its own price
history, deciding each day whether to hold a long, flat, or short position. This is
Part 2 of a two-part reinforcement learning project: Part 1 planned the agent and fixed
a success bar in advance, and Part 2 built it, trained it, and tested it out of sample
the way the plan required. The agent is trained and tested on SPY (the S&P 500 ETF) and
the whole pipeline is rerun on QQQ (the Nasdaq-100 ETF) as a second instrument.

**Headline result: from catastrophe to a competitive strategy.** The naive agent built
to the Part 1 plan failed badly out of sample: on the 2021–2023 test period its mean
Sharpe ratio across ten seeds was **−0.98**, worse than a random policy (−1.01), with a
47% drawdown, because it shorted an upward-drifting index and overtraded. Diagnosing that
on the validation data led to a redesigned agent (long-only, a turnover penalty, a
20-seed majority-vote ensemble, and a configuration **selected for insensitivity to
transaction costs**) that lifted the Sharpe ratio to **+0.52** and became a low-turnover
trend-follower: about 85% long, ~12 trades a year, cost-insensitive, earning ~7%/yr
(roughly two-thirds of the market) at a similar drawdown. It **matches buy-and-hold on
SPY** (+0.65) and **beats it out of sample on QQQ** (+0.67 vs +0.54). It still does not
clear the strict pre-set bar on SPY (buy-and-hold edges it by ~2 SE) and a once-a-year
moving-average rule (+1.06) stays the single best strategy, which keeps the result honest.
The lesson is twofold: deep RL can be engineered from a catastrophic failure into a
competitive strategy, but the marginal edge over a passive index is small and hard-won,
and a leakage-free, cost-aware test is what earns the improvement without ever tuning on
the test set.

## What this does

Implements a **Double-DQN** trading agent (Mnih et al., 2015; van Hasselt et al., 2016)
in a custom Gymnasium environment. The agent sees a 60-day window of returns, a few
technical features, and its current position. A naive version picks short, flat, or
long with a raw log-return reward; the improved version is long-only, adds a turnover
penalty and a risk-adjusted differential-Sharpe option, and deploys a 20-seed
majority-vote ensemble. Every change was diagnosed and selected on the validation block.

The point of the project is not the agent but the **evaluation**, which is built to
resist overfitting:

- **Time-ordered split, never shuffled.** Train 2010–2018, validate 2019–2020, test
  2021–2023, with a 2005–2009 crisis window held back.
- **No leakage.** Every feature and every normalization statistic is computed on
  training data only and then frozen.
- **Honest scoring.** Ten random seeds, costs on every trade, four baselines
  (buy-and-hold, all-cash, random, moving-average crossover), a majority-vote ensemble,
  and a success criterion fixed before the test set was ever touched.

## Quick start

```bash
pip install -r requirements.txt         # PyTorch, Gymnasium, pandas, matplotlib (+ python-pptx for the deck)
python3 src/data.py                     # download SPY and QQQ daily prices to data/raw/
python3 src/experiments.py --stage all  # regularization, grid, naive test, stress
python3 src/experiments.py --stage improve  # the long-only, ensembled, improved agent
python3 src/plots.py                    # redraw the figures from outputs/
python3 src/demo.py                     # render the demo video (needs ffmpeg)
```

`src/experiments.py` runs the stages in order and writes every result to
`outputs/` and the per-seed checkpoints to `models/`. The ten-seed final training takes
about half a minute on an Apple M5 Pro; the full suite, a few minutes. `build.sh`
rebuilds the figures, the demo, and the slide deck from the saved outputs.

## Layout

```
├── src/
│   ├── data.py         # Yahoo Finance download, causal features, time-ordered splits, train-only scaling
│   ├── env.py          # Gymnasium env (short/flat/long or long-only, costs, turnover penalty, log-return/Sharpe rewards)
│   ├── agent.py        # Q-network, replay buffer, and the Double-DQN update
│   ├── train.py        # one training run with validation checkpointing and full logging
│   ├── evaluate.py     # rolling a policy, the four baselines, trading metrics, the success test
│   ├── experiments.py  # stages: regularization, grid, naive test, stress, and the improved agent; run in parallel
│   ├── plots.py        # the five figures, drawn from outputs/
│   ├── demo.py         # the recorded demo video (improved agent trading the test period vs buy-and-hold)
│   └── build_deck.py   # the presentation, built on deck_style.py
├── figures/            # fig01 equity (naive vs improved) · fig02 recovery · fig03 training · fig04 positions · fig05 cost stress
├── outputs/            # metrics (csv), success tests (json), per-seed test runs (npz), training logs, run log
├── data/raw/           # SOURCE.txt provenance (the downloaded CSVs are not tracked)
├── build.sh
├── requirements.txt
└── README.md
```

Model checkpoints (`models/`), the downloaded price CSVs, and the large binary
deliverables are not tracked: they are regenerable from the code and the pinned
settings.

## What the agent learns (and does not)

Trained for 200 one-year episodes per seed with the settings in `src/train.py`
(Double-DQN, γ = 0.9, lr = 1e-4, 60-day window, a 64-unit network with weight decay and
observation noise, 5 bps cost):

- **Training works, generalization does not.** The episode reward and the mean
  predicted Q-value rise and plateau, but the validation Sharpe ratio stays noisy and
  never settles: the network optimizes its objective without finding a transferable
  signal.
- **The naive agent loses.** Mean test Sharpe −0.98 (−16%/yr, 47% drawdown) against
  buy-and-hold at +0.65; a random policy scores −1.01, so it trades no better than chance.
- **The redesign makes it competitive.** Long-only, turnover-penalized, ensembled, and
  selected (on validation) to be insensitive to costs, the improved agent reaches +0.52
  on SPY: a low-turnover trend-follower, ~85% long, ~7%/yr at a 27% drawdown, no shorting.
- **It matches SPY and beats QQQ.** It comes within ~2 SE of buy-and-hold on SPY (+0.52
  vs +0.65) and its ensemble beats buy-and-hold out of sample on QQQ (+0.67 vs +0.54),
  while staying cost-insensitive; a once-a-year moving-average rule (+1.06) still leads.

`figures/fig01_test_equity.png` shows the naive and improved equity curves against the
baselines; `fig02_recovery.png` ranks the strategies by Sharpe and drawdown;
`fig03_training.png` shows the reward and Q-value climbing while validation stays flat.

## Note on the written report and the presentation

The written report, the slide deck, the narration, the recorded demo, and the reference
PDFs are kept private and are intentionally **not** part of this repository (see
`.gitignore`). This repo tracks only the code and the reproducible run artifacts
(figures and outputs), which regenerate all of them.
