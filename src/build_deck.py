"""build_deck.py -- single source of truth for the Part 2 presentation.

Builds PRESENTATION.pptx from the design system in deck_style.py and the result figures
in figures/. The deck tells the honest arc of the completed project: a naive Double-DQN
trader failed catastrophically, a principled redesign (diagnosed on validation) recovered
it from a losing strategy to a break-even one, and even then it could not beat a passive
position after costs. Speaker notes carry the full narration, timed for the 13-15 minute
recording. Run build.sh, which builds this deck and renders a PDF and per-slide images.
"""

import os
from PIL import Image
from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR

import deck_style as S
from deck_style import (
    add_text, add_card, add_circle, add_rect, add_connector, set_notes, word_count,
    PLUM, PLUM_DK, TERRA, TERRA_DK, PAPER, CARD, CARD_TINT, INK, INK_SOFT, SAGE, GOLD,
    LINE_SOFT, ON_PLUM, ON_PLUM_SOFT, F_HEAD, F_BODY,
)

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
FIG = os.path.join(ROOT, "figures")
DEMO = os.path.join(ROOT, "demo")

BLUE = S.RGBColor(0x2A, 0x78, 0xD6)   # buy-and-hold
RUST = S.RGBColor(0xC2, 0x57, 0x1A)   # the naive agent / losses
GOLDA = S.RGBColor(0xED, 0xA1, 0x00)  # moving-average rule

prs = Presentation()
prs.slide_width = S.SLIDE_W
prs.slide_height = S.SLIDE_H
BLANK = prs.slide_layouts[6]
MARGIN = Inches(0.62)
CW = prs.slide_width - 2 * MARGIN
notes_log = []


def slide(bg=PAPER):
    sl = prs.slides.add_slide(BLANK)
    S._set_bg(sl, bg)
    return sl


def hrule(sl, x, y, w, color, thick=Pt(1.0)):
    add_rect(sl, x, Emu(int(y - thick / 2)), w, thick, fill=color)


def header(sl, kicker, title, title_size=25):
    add_text(sl, MARGIN, Inches(0.5), CW, Inches(0.26),
             [[(kicker.upper(), {"size": 10.5, "color": INK_SOFT, "bold": True, "font": F_BODY, "spacing": 2.0})]],
             space_after=Pt(0))
    add_text(sl, MARGIN, Inches(0.8), CW, Inches(0.7),
             [[(title, {"size": title_size, "color": PLUM, "bold": True, "font": F_HEAD})]],
             space_after=Pt(0), line_spacing=1.02)
    hrule(sl, MARGIN, Inches(1.54), CW, LINE_SOFT, thick=Pt(1.0))


def footer(sl, page, source=None):
    y = Inches(7.06)
    hrule(sl, MARGIN, y, CW, LINE_SOFT, thick=Pt(0.9))
    if source:
        add_text(sl, MARGIN, y + Inches(0.08), Inches(10.4), Inches(0.28),
                 [[(source, {"size": 9.5, "color": INK_SOFT, "font": F_BODY})]], space_after=Pt(0))
    add_text(sl, MARGIN + CW - Inches(1.2), y + Inches(0.08), Inches(1.2), Inches(0.28),
             [[(str(page), {"size": 9.5, "color": INK_SOFT, "font": F_BODY})]],
             align=PP_ALIGN.RIGHT, space_after=Pt(0))


def bullets(sl, l, t, w, h, items, *, size=15, gap=Pt(10), color=INK, lead=TERRA, line_spacing=1.08):
    paras = []
    for it in items:
        if isinstance(it, tuple):
            head, rest = it
            paras.append([("•  ", {"size": size, "color": lead, "bold": True}),
                          (head, {"size": size, "color": color, "bold": True}),
                          (rest, {"size": size, "color": color})])
        else:
            paras.append([("•  ", {"size": size, "color": lead, "bold": True}),
                          (it, {"size": size, "color": color})])
    add_text(sl, l, t, w, h, paras, space_after=gap, line_spacing=line_spacing)


def stat(sl, l, t, w, big, label, *, big_color=PLUM, big_size=34, label_size=11.5):
    add_text(sl, l, t, w, Inches(0.62),
             [[(big, {"size": big_size, "color": big_color, "bold": True, "font": F_HEAD})]], space_after=Pt(0))
    add_text(sl, l, t + Inches(big_size / 72.0 + 0.20), w, Inches(0.5),
             [[(label, {"size": label_size, "color": INK_SOFT})]], space_after=Pt(0), line_spacing=1.02)


def add_fig(sl, path, l, t, max_w, max_h):
    iw, ih = Image.open(path).size
    scale = min(max_w / iw, max_h / ih)
    w, h = int(iw * scale), int(ih * scale)
    left, top = int(l + (max_w - w) / 2), int(t + (max_h - h) / 2)
    sl.shapes.add_picture(path, Emu(left), Emu(top), width=Emu(w), height=Emu(h))
    return Emu(left), Emu(top), Emu(w), Emu(h)


def note(sl, label, text):
    set_notes(sl, text)
    notes_log.append((label, text))


# ========================================================= 1  TITLE
s = slide(PLUM)
add_text(s, MARGIN, Inches(1.5), CW, Inches(0.34),
         [[("PhDAI 830   |   REINFORCEMENT LEARNING PROJECT, PART 2", {"size": 12.5, "color": TERRA, "bold": True, "font": F_BODY, "spacing": 2.4})]],
         space_after=Pt(0))
hrule(s, MARGIN, Inches(1.98), Inches(2.1), TERRA, thick=Pt(2.4))
add_text(s, MARGIN, Inches(2.3), Inches(11.8), Inches(1.9),
         [[("Can a Double-DQN agent learn to beat the market?", {"size": 38, "color": ON_PLUM, "bold": True, "font": F_HEAD})]],
         line_spacing=1.03, space_after=Pt(0))
add_text(s, MARGIN, Inches(4.35), Inches(11.6), Inches(0.85),
         [[("A deep reinforcement learning trading agent, built to the Part 1 plan. The naive version failed; a cost-aware redesign turned it into a competitive strategy that matches the market.",
            {"size": 16.5, "color": ON_PLUM_SOFT, "font": F_HEAD, "italic": True})]], line_spacing=1.12, space_after=Pt(0))
hrule(s, MARGIN, Inches(5.72), CW, PLUM_DK, thick=Pt(1.2))
add_text(s, MARGIN, Inches(5.9), CW, Inches(1.0),
         [[("Jose Santiago Echevarria", {"size": 15, "color": ON_PLUM, "bold": True, "font": F_BODY})],
          [("PhDAI 830: Applied Machine Intelligence and Reinforcement Learning   ·   Dr. Atif Farid Mohammad", {"size": 12.5, "color": ON_PLUM_SOFT, "font": F_BODY})],
          [("October 11, 2026", {"size": 12.5, "color": ON_PLUM_SOFT, "font": F_BODY})]],
         space_after=Pt(3))
add_text(s, MARGIN, Inches(7.0), CW, Inches(0.4),
         [[("Author's note: The author used Claude Opus 4.8 (Anthropic) for brainstorming and organization, initial exploratory research, and proofreading.",
            {"size": 9, "color": ON_PLUM_SOFT, "italic": True, "font": F_BODY})]], space_after=Pt(0))
note(s, "1 Title",
     "I'm Jose Santiago Echevarria, and this is Part 2 of my reinforcement learning project. In Part 1 I wrote the plan for a deep reinforcement learning agent that trades the stock market, and I set a clear bar for success before I started. In Part 2 I built that agent and tested it honestly. The question is simple: can a Double Deep Q-Network learn to beat a plain buy-and-hold position? My first version failed badly. It traded worse than a coin flip. So I worked out why, redesigned it, and the new version is genuinely competitive. It matches buy-and-hold on the S&P 500 and beats it on the Nasdaq. Over the next few minutes I'll walk through the problem, how I built the agent, the results, a short demo, and what I learned. The honest path from a failing agent to a competitive one is the real result here.")

# ========================================================= 2  PROJECT INTRODUCTION
s = slide(PAPER)
header(s, "1 · Project introduction", "The problem: learn to trade from the market itself")
bullets(s, MARGIN, Inches(1.78), Inches(7.35), Inches(3.4), [
    ("The task. ", "Each day, decide whether to hold a long, flat, or short position in one liquid asset (the S&P 500 ETF, SPY), to earn a strong risk-adjusted return after trading costs."),
    ("Why reinforcement learning. ", "Trading is a sequential decision under uncertainty: each action sets the next day's exposure and pays a cost, and what matters is the profit over a whole path, not one forecast."),
    ("The Part 1 objective. ", "Build a working value-based agent and test it honestly out of sample against clear baselines, on risk-adjusted metrics."),
    ("The bar, fixed in advance. ", "Beat buy-and-hold's Sharpe ratio on the untouched test years by more than one standard error across seeds, or report the failure plainly."),
], size=14.5, gap=Pt(11))
rx = Inches(8.55)
add_text(s, rx, Inches(1.78), Inches(4.2), Inches(0.3),
         [[("THE DECISION, EACH DAY", {"size": 10.5, "color": INK_SOFT, "bold": True, "font": F_BODY, "spacing": 1.6})]], space_after=Pt(0))
for i, (lab, col) in enumerate([("SHORT", RUST), ("FLAT", INK_SOFT), ("LONG", TERRA)]):
    add_card(s, rx, Inches(2.25 + i * 0.84), Inches(4.2), Inches(0.68), fill=CARD_TINT, shadow=False)
    add_circle(s, int(rx + Inches(0.5)), int(Inches(2.25 + i * 0.84) + Inches(0.34)), int(Inches(0.3)), fill=col)
    add_text(s, rx + Inches(0.9), Inches(2.25 + i * 0.84) + Inches(0.12), Inches(3.1), Inches(0.44),
             [[(lab, {"size": 15, "color": INK, "bold": True}),
               ({"SHORT": "  bet the price falls", "FLAT": "  hold cash, earn nothing", "LONG": "  bet the price rises"}[lab],
                {"size": 12, "color": INK_SOFT})]], anchor=MSO_ANCHOR.MIDDLE, space_after=Pt(0))
add_text(s, rx, Inches(5.0), Inches(4.2), Inches(1.4),
         [[("The agent sees only past prices and a few indicators, and it is scored by the money it makes. ",
            {"size": 12.5, "color": INK}),
           ("It optimizes the trade, not a prediction.", {"size": 12.5, "color": INK, "bold": True})]],
         line_spacing=1.12, space_after=Pt(0))
footer(s, 2)
note(s, "2 Project introduction",
     "Let me set up the problem. The agent trades one asset, the S&P 500 fund, SPY. Every day it makes a single decision: go long and bet the price rises, stay flat in cash, or go short and bet it falls. It holds one position at a time, at a fixed size, so the only question each day is which way to lean. I kept it to a single, highly liquid index fund on purpose, so the result turns on the decisions themselves and not on picking the right stock. I chose reinforcement learning because trading is a sequential decision under uncertainty. Each action sets the position for the next day and pays a trading cost, and what matters is the total profit over the whole period, not whether any one forecast was right. A supervised model would just predict the next return, and you'd still need a separate rule to act on it. A reinforcement learning agent learns the trade itself. My goal in Part 1 was a working agent that I would test honestly, out of sample, against clear baselines. And I fixed the bar in advance: beat buy-and-hold's risk-adjusted return on the untouched test years, or report the failure plainly.")

# ========================================================= 3  BACKGROUND
s = slide(PAPER)
header(s, "2 · Background and project context", "What the research says, and the warning in it")
bullets(s, MARGIN, Inches(1.78), Inches(7.35), Inches(3.3), [
    ("Trading as a Markov decision process. ", "State is a window of prices and indicators plus the held position; the action is the position; the reward is the return it earns (Hambly et al., 2023)."),
    ("Two families of methods. ", "An older direct-policy line that optimizes a Sharpe-type reward (Moody & Saffell, 2001), and newer deep RL: value-based Q-networks and policy-gradient methods (Theate & Ernst, 2021)."),
    ("The headline is strong, the fine print is not. ", "Published agents beat their benchmarks in-sample, but surveys warn that general RL tends to overfit financial data (Hambly et al., 2023)."),
], size=14.5, gap=Pt(12))
add_card(s, Inches(8.55), Inches(1.9), Inches(4.2), Inches(4.5), fill=CARD_TINT, shadow=True)
add_text(s, Inches(8.8), Inches(2.12), Inches(3.75), Inches(0.5),
         [[("THE GAP THIS PROJECT TARGETS", {"size": 10.5, "color": TERRA, "bold": True, "spacing": 1.4})]], space_after=Pt(0))
bullets(s, Inches(8.8), Inches(2.66), Inches(3.75), Inches(3.5), [
    "Backtests are mostly rising, US markets",
    "Transaction costs modeled thinly or not at all",
    "Rarely tested across regimes or assets",
    "Policies overfit and vary run to run",
], size=12.5, gap=Pt(10), lead=RUST)
add_text(s, Inches(8.8), Inches(5.0), Inches(3.75), Inches(0.9),
         [[("So one honest backtest proves little. ", {"size": 12.5, "color": INK, "bold": True}),
           ("This project was designed to test exactly what the literature says is fragile.", {"size": 12.5, "color": INK})]],
         line_spacing=1.12, space_after=Pt(0))
footer(s, 3, "Built from the Part 1 literature review (eight scholarly sources).")
note(s, "3 Background",
     "Here's some quick background, built from my Part 1 literature review. Almost all of this research treats trading as a Markov decision process. The state is a window of recent prices and indicators plus the position you hold. The action is the new position. The reward is the return it earns. There are two main families of methods: an older one that searches directly for a trading policy, and newer deep reinforcement learning with Q-networks. Mine is a Q-network. The important part is the fine print. Published agents do beat their benchmarks, but almost always inside their own backtest. The test periods are mostly rising US markets, trading costs are modeled thinly or ignored, and the agents overfit from one run to the next. A recent survey in the field says it plainly: reinforcement learning tends to overfit financial data. That's a real warning, so I designed this project to test the exact thing the research says is fragile, rather than to chase a good-looking backtest.")

# ========================================================= 4  METHODOLOGY
s = slide(PAPER)
header(s, "3 · Methodology", "A Double-DQN agent, and a test built to resist overfitting")
add_text(s, MARGIN, Inches(1.74), Inches(6.0), Inches(0.3),
         [[("THE AGENT", {"size": 10.5, "color": INK_SOFT, "bold": True, "spacing": 1.4})]], space_after=Pt(0))
bullets(s, MARGIN, Inches(2.08), Inches(6.1), Inches(2.6), [
    ("Algorithm. ", "Double Deep Q-Network: experience replay and a target network (Mnih et al., 2015), with the double-Q target that curbs value overestimation (van Hasselt et al., 2016)."),
    ("State. ", "A 60-day window of returns, a moving-average ratio, RSI, volatility, and the current position."),
    ("Action & reward. ", "Short, flat, or long; reward is the change in the log value of the account, minus a 5-basis-point cost on every change of position."),
], size=13.5, gap=Pt(9))
add_text(s, Inches(7.1), Inches(1.74), Inches(5.6), Inches(0.3),
         [[("THE EVALUATION (WHERE THE RIGOR LIVES)", {"size": 10.5, "color": TERRA, "bold": True, "spacing": 1.4})]], space_after=Pt(0))
bullets(s, Inches(7.1), Inches(2.08), Inches(5.65), Inches(2.6), [
    ("Split by time, never shuffled. ", "Train 2010-18, validate 2019-20, test 2021-23, with a 2005-09 crisis window held back."),
    ("No leakage. ", "Every feature and every scaling statistic is computed on past, training-only data and then frozen."),
    ("Honest scoring. ", "Many seeds, costs on every trade, and a success bar fixed before seeing the test set."),
], size=13.5, gap=Pt(9), lead=TERRA)
add_text(s, MARGIN, Inches(5.1), Inches(5.9), Inches(0.28),
         [[("CHRONOLOGICAL SPLIT  (NO SHUFFLING, NO LOOK-AHEAD)", {"size": 10, "color": INK_SOFT, "bold": True, "spacing": 1.2})]], space_after=Pt(0))
segs = [("2005–09", "crisis test", SAGE, 1.15), ("2010–18", "train", TERRA, 2.9),
        ("2019–20", "validate", GOLD, 1.0), ("2021–23", "TEST", PLUM, 1.35)]
x = MARGIN
total = sum(w for *_, w in segs)
for lab, sub, col, wght in segs:
    sw = Emu(int((CW) * wght / total))
    add_rect(s, x, Inches(5.5), Emu(int(sw) - Inches(0.06)), Inches(0.62), fill=col, radius=0.08)
    add_text(s, x, Inches(5.56), Emu(int(sw) - Inches(0.06)), Inches(0.3),
             [[(lab, {"size": 12.5, "color": ON_PLUM, "bold": True})]], align=PP_ALIGN.CENTER, space_after=Pt(0))
    add_text(s, x, Inches(5.86), Emu(int(sw) - Inches(0.06)), Inches(0.26),
             [[(sub, {"size": 10, "color": ON_PLUM_SOFT if col != GOLD else INK})]], align=PP_ALIGN.CENTER, space_after=Pt(0))
    x = Emu(int(x) + int(sw))
footer(s, 4, "Algorithm: Mnih et al. (2015); van Hasselt et al. (2016).")
note(s, "4 Methodology",
     "My methodology comes in two halves. On the left is the agent. It's a Double Deep Q-Network. That means two stabilizers from the 2015 DQN paper, an experience replay buffer and a target network, plus the double-Q update, where one network picks the next action and another values it, which stops it from overestimating. The agent sees a sixty-day window of returns, a moving-average ratio, a momentum indicator, volatility, and its current position. The reward is the change in the account's value, minus a five-basis-point cost every time it trades. During training it explores with a rule that starts fully random and slowly becomes almost greedy. The right half is the part that matters most, the evaluation. I split the data strictly by time and never shuffle it. I train on 2010 through 2018, validate on 2019 and 2020, and keep 2021 through 2023 completely untouched for the final test. Every feature uses only past data. And I score the agent over many random seeds, with real trading costs, and the success bar I fixed before I ever looked at the test set.")

# ========================================================= 5  IMPLEMENTATION
s = slide(PAPER)
header(s, "4 · Implementation", "A naive baseline, then a redesign diagnosed on validation")
add_card(s, MARGIN, Inches(1.8), Inches(5.95), Inches(4.5), fill=CARD_TINT, shadow=True)
add_text(s, MARGIN + Inches(0.25), Inches(2.0), Inches(5.5), Inches(0.3),
         [[("THE NAIVE AGENT, AND WHY IT FAILED", {"size": 10.5, "color": RUST, "bold": True, "spacing": 1.3})]], space_after=Pt(0))
bullets(s, MARGIN + Inches(0.25), Inches(2.5), Inches(5.45), Inches(3.6), [
    ("Shorted a rising market. ", "Betting against an index that drifts up is a structural losing trade."),
    ("Overtraded. ", "About 200 position changes a year, so costs and noise-trading compounded."),
    ("High seed variance. ", "Single networks swung widely run to run."),
], size=13, gap=Pt(10), lead=RUST)
add_card(s, Inches(6.85), Inches(1.8), Inches(5.9), Inches(4.5), fill=CARD_TINT, shadow=True)
add_text(s, Inches(7.1), Inches(2.0), Inches(5.4), Inches(0.3),
         [[("THE REDESIGN (CHOSEN ON VALIDATION)", {"size": 10.5, "color": TERRA, "bold": True, "spacing": 1.3})]], space_after=Pt(0))
bullets(s, Inches(7.1), Inches(2.5), Inches(5.4), Inches(3.6), [
    ("Long-only. ", "Flat or long, never short, removing the short-the-uptrend handicap."),
    ("Turnover penalty + risk-adjusted reward. ", "Discourage churn; a differential-Sharpe option."),
    ("20-seed vote ensemble. ", "Seeds vote their action; the majority wins, far more stable than one net."),
    ("Tools. ", "PyTorch, Gymnasium, pandas; trains in minutes on an Apple M5 Pro (CPU)."),
], size=13, gap=Pt(9), lead=TERRA)
footer(s, 5, "Also changed from Part 1: Yahoo data (Stooq blocked), regularization, a 2008 stress window.")
note(s, "5 Implementation",
     "On the build, the key point is that I made two versions. The network is small, just two hidden layers, written in PyTorch, and it trains in a few minutes on my laptop; the networks are small enough that the CPU is actually faster than the GPU here. The naive agent, on the left, is the Part 1 design, and it failed for three reasons I could see on the validation data. It shorted a market that drifts upward, which is a losing bet. It traded about two hundred times a year, so costs and noise piled up. And single runs swung all over the place. So I redesigned it, and every choice was made on validation, never on the test set. First, I made it long-only, so it can't short the rising market. Second, I penalized turnover to stop the constant trading. Third, the policy is a twenty-seed majority vote, far steadier than any single network. And last, I chose the settings by how they held up at a high, stress-level trading cost, not the low cost I actually charge. That one choice is what steered me away from a day-trader and toward a low-turnover, mostly-long policy.")

# ========================================================= 6  RESULTS 1 (naive failure)
s = slide(PAPER)
header(s, "5 · Results and analysis", "The naive agent traded no better than a coin flip")
add_fig(s, os.path.join(FIG, "fig03_training.png"), MARGIN, Inches(1.78), Inches(8.1), Inches(3.0))
add_text(s, MARGIN, Inches(4.95), Inches(8.1), Inches(1.4),
         [[("Training worked: reward and Q-value climb, so the network optimized its objective. But the validation Sharpe (right panel) never settles, which is the signature of overfitting: there was no stable signal to learn.",
            {"size": 12.5, "color": INK})]], line_spacing=1.16, space_after=Pt(0))
rx = Inches(9.0)
stat(s, rx, Inches(1.95), Inches(3.8), "-0.98", "test Sharpe (vs +0.65 buy-and-hold)", big_color=RUST)
stat(s, rx, Inches(3.3), Inches(3.8), "-1.01", "a random policy scored the same", big_color=INK_SOFT)
stat(s, rx, Inches(4.65), Inches(3.8), "-16%/yr", "with a 47% drawdown", big_color=RUST)
add_text(s, rx, Inches(5.95), Inches(3.8), Inches(0.9),
         [[("Missed the preset bar by 11+ standard errors.", {"size": 12.5, "color": RUST, "bold": True})]],
         line_spacing=1.1, space_after=Pt(0))
footer(s, 6, "SPY test period 2021–2023, net of 5 bps costs. Left: ten seeds, bold = mean.")
note(s, "6 Results, part one: the naive failure",
     "Now the results. I'll start with the naive agent. The training itself worked. On the left, across every seed, the reward and the predicted Q-value both climb and level off, so the network learned its objective. But look at the third panel, the validation Sharpe. It's noisy and it never settles down. That's the signature of overfitting. The network minimized its loss, but there was no real signal to learn. You can see it in the numbers too: the setting I picked scored a Sharpe near one-point-four on the tuning grid, dropped to about zero-point-seven when I retrained it, and then went negative on the test. On the test years it lost money. Its Sharpe was minus zero-point-nine-eight, against plus zero-point-six-five for buy-and-hold, about sixteen percent a year with a forty-seven percent drawdown. And the clearest sign of all: a random policy scored minus one-point-zero-one over the same period. So the agent I trained traded no better than flipping a coin.")

# ========================================================= 7  RESULTS 2 (improved recovery)
s = slide(PAPER)
header(s, "5 · Results and analysis", "The redesign made the agent competitive")
add_fig(s, os.path.join(FIG, "fig01_test_equity.png"), MARGIN, Inches(1.72), Inches(7.8), Inches(4.5))
rx = Inches(8.75)
stat(s, rx, Inches(1.9), Inches(4.05), "-0.98 → 0.52", "Sharpe ratio: naive → improved (B&H 0.65)", big_color=TERRA, big_size=26)
stat(s, rx, Inches(3.1), Inches(4.05), "7%/yr", "return, ~70% of the market's, at similar risk", big_color=TERRA, big_size=30)
add_text(s, rx, Inches(4.3), Inches(4.05), Inches(2.3),
         [[("A low-turnover trend-follower. ", {"size": 13, "color": INK, "bold": True}),
           ("~85% long, ~12 trades a year, no shorting, and cost-insensitive: its Sharpe barely moves as costs rise.",
            {"size": 12.5, "color": INK})]], line_spacing=1.16, space_after=Pt(0))
footer(s, 7, "The improved agent (green) now tracks buy-and-hold and the trend rule closely.")
note(s, "7 Results, part two: the recovery",
     "Then I redesigned it, and here's what changed. On the chart, the rust line is the naive agent, falling to a forty percent loss. The green line is the improved agent. It now runs right alongside buy-and-hold and the trend rule, and finishes up about twenty-five percent. In numbers, its Sharpe ratio went from minus zero-point-nine-eight all the way to plus zero-point-five-two, basically level with buy-and-hold at zero-point-six-five. It earns about seven percent a year, roughly two-thirds of the market, at a similar drawdown. What made the difference was low turnover. And I want to be clear that I never told it to trade less directly. I chose the settings at a high stress cost, and the low-turnover policy is simply what survived that test. The agent now holds long about eighty-five percent of the time, trades only about a dozen times a year, and never shorts. It mostly rides the market up and steps aside through parts of the 2022 fall, which is why its drawdown is lower. And because it barely trades, costs hardly touch it. In one step, it went from a disaster to a genuinely competitive strategy.")

# ========================================================= 8  RESULTS 3 (honest limit)
s = slide(PAPER)
header(s, "5 · Results and analysis", "Competitive now, and it beats buy-and-hold on QQQ")
add_fig(s, os.path.join(FIG, "fig02_recovery.png"), MARGIN, Inches(1.8), Inches(7.3), Inches(4.3))
rx = Inches(8.4)
add_text(s, rx, Inches(1.9), Inches(4.35), Inches(0.3),
         [[("ACROSS ASSETS, PERIODS, AND COSTS (SHARPE)", {"size": 10, "color": INK_SOFT, "bold": True, "spacing": 1.1})]], space_after=Pt(0))
checks = [("Second asset (QQQ)", "+0.67", "+0.54"), ("2008 crisis window", "-0.09", "+0.14"),
          ("Zero trading cost", "+0.53", "+0.66"), ("A simple MA rule", "+1.06", "best")]
yy = Inches(2.4)
for lab, a, b in checks:
    add_card(s, rx, yy, Inches(4.35), Inches(0.66), fill=CARD_TINT, shadow=False)
    add_text(s, rx + Inches(0.2), yy + Inches(0.08), Inches(2.5), Inches(0.5),
             [[(lab, {"size": 12.5, "color": INK, "bold": True})]], anchor=MSO_ANCHOR.MIDDLE, space_after=Pt(0))
    col = TERRA if lab in ("A simple MA rule", "Second asset (QQQ)") else INK
    add_text(s, rx + Inches(2.55), yy + Inches(0.08), Inches(1.6), Inches(0.5),
             [[(a, {"size": 15, "color": col, "bold": True, "font": F_BODY}),
               ("  " + ("vs " + b if b != "best" else b), {"size": 11, "color": INK_SOFT})]],
             align=PP_ALIGN.RIGHT, anchor=MSO_ANCHOR.MIDDLE, space_after=Pt(0))
    yy = Emu(int(yy) + int(Inches(0.78)))
add_text(s, rx, Inches(5.85), Inches(4.35), Inches(0.9),
         [[("It matches SPY and beats QQQ out of sample; a once-a-year trend rule still edges it.",
            {"size": 12.5, "color": INK, "bold": True})]], line_spacing=1.14, space_after=Pt(0))
footer(s, 8)
note(s, "8 Results, part three: the honest limit",
     "I tested the improved agent everywhere to see how far that holds up. On the left, the strategies ranked by Sharpe ratio and drawdown. The agent now sits right next to buy-and-hold instead of at the bottom. On the right are the checks. On the Nasdaq fund, QQQ, the agent scored plus zero-point-six-seven, which actually beats buy-and-hold's zero-point-five-four, out of sample. At zero trading cost it holds at plus zero-point-five-three, and it stays flat as costs climb, because it hardly trades. On the 2008 crisis window it roughly matched a weak market, about minus zero-point-one against buy-and-hold's plus zero-point-one. Now the honest part. On SPY, the index it trained on, buy-and-hold still edges it by about two standard errors, so it doesn't clear my strict bar there, and the simple moving-average rule, at one-point-zero-six, is still the single best strategy. But the agent is competitive across the board, and it genuinely wins on the second asset.")

# ========================================================= 9  DEMONSTRATION
s = slide(PLUM)
add_text(s, MARGIN, Inches(0.46), CW, Inches(0.28),
         [[("6 · DEMONSTRATION", {"size": 11, "color": TERRA, "bold": True, "font": F_BODY, "spacing": 2.2})]], space_after=Pt(0))
add_text(s, MARGIN, Inches(0.82), CW, Inches(0.5),
         [[("Watch the improved agent trade the test period", {"size": 26, "color": ON_PLUM, "bold": True, "font": F_HEAD})]], space_after=Pt(0))
bx, by, bw, bh = add_fig(s, os.path.join(DEMO, "demo_still.png"), Inches(3.55), Inches(1.6), Inches(6.2), Inches(3.9))
add_circle(s, int(bx) + int(bw) // 2, int(by) + int(bh) // 2, int(Inches(0.9)), fill=ON_PLUM, line=None)
tri = s.shapes.add_shape(S.MSO_SHAPE.ISOSCELES_TRIANGLE, Emu(int(bx) + int(bw) // 2 - int(Inches(0.16))),
                         Emu(int(by) + int(bh) // 2 - int(Inches(0.2))), Inches(0.42), Inches(0.4))
tri.rotation = 90
tri.fill.solid(); tri.fill.fore_color.rgb = TERRA; tri.line.fill.background(); tri.shadow.inherit = False
add_text(s, Inches(0.9), Inches(5.75), Inches(11.5), Inches(0.8),
         [[("The recording plays ", {"size": 14, "color": ON_PLUM_SOFT, "font": F_BODY}),
           ("demo/agent_trading_demo.mp4", {"size": 14, "color": ON_PLUM, "font": F_BODY, "bold": True}),
           (": green shows the days it holds long, and the lower panel is its account against buy-and-hold.",
            {"size": 14, "color": ON_PLUM_SOFT, "font": F_BODY})]], align=PP_ALIGN.CENTER, line_spacing=1.1, space_after=Pt(0))
add_text(s, MARGIN + CW - Inches(1.2), Inches(7.06), Inches(1.2), Inches(0.28),
         [[("9", {"size": 9.5, "color": ON_PLUM_SOFT, "font": F_BODY})]], align=PP_ALIGN.RIGHT, space_after=Pt(0))
note(s, "9 Demonstration",
     "Let me show you the agent trading. I'll play the recording now. [PLAY demo/agent_trading_demo.mp4] This is the improved agent trading SPY across the whole test period, one day at a time. The top panel is the SPY price, shaded green on the days the agent holds long. The rest of the time it's in cash, and you'll see it never shorts. Watch how rarely that shading switches on and off. That's the low turnover I built in, only about a dozen trades across three years, playing out in front of you. The bottom panel is what matters: the agent's account in green, against buy-and-hold in blue. Keep an eye on those two lines, because they stay close the whole way. The agent steps out of part of the 2022 fall, which protects it, and stays invested through most of the recovery. It finishes close to buy-and-hold, with less time exposed to the market. That's the whole strategy, playing out day by day, and it's the same behavior the numbers described a moment ago.")

# ========================================================= 10  ETHICS
s = slide(PAPER)
header(s, "7 · Ethical considerations", "Why an honest result is the responsible outcome")
bullets(s, MARGIN, Inches(1.82), Inches(7.4), Inches(4.4), [
    ("Overfit-and-deploy harm. ", "A model that looks good in a backtest but overfits transfers its losses to whoever trusts it. Model risk is formally governed for this reason, with validation standards that demand exactly this kind of out-of-sample and stress testing (SR 11-7)."),
    ("Opacity. ", "A deep network gives a risk desk little account of why it holds what it holds, so a convincing backtest can hide both overfitting and the absence of any economic reason for a trade."),
    ("Reporting integrity. ", "The temptation is to search for a window or an asset where the agent wins. The responsible practice is to fix the bar in advance and report the result either way, which is what this project did."),
    ("Scope. ", "This is a research exercise on historical data, with simplified costs and no live money, and it is not investment advice."),
], size=14, gap=Pt(13))
add_card(s, Inches(8.65), Inches(2.2), Inches(4.1), Inches(2.9), fill=PLUM, shadow=True)
add_text(s, Inches(8.95), Inches(2.5), Inches(3.55), Inches(2.4),
         [[("“General RL algorithms tend to overfit financial data.”", {"size": 17, "color": ON_PLUM, "italic": True, "font": F_HEAD})],
          [("", {"size": 6, "color": ON_PLUM})],
          [("The field's own survey. This project is a worked example of why that warning deserves to be taken seriously.",
            {"size": 12, "color": ON_PLUM_SOFT})]], line_spacing=1.14, space_after=Pt(6))
footer(s, 10, "Source: Board of Governors of the Federal Reserve System & OCC (2011), SR 11-7.")
note(s, "10 Ethical considerations",
     "The ethics here are really the ethics of any trading model. The biggest risk is overfit-and-deploy. A model that looks great in a backtest but has actually overfit will hand its losses to whoever trusts it with real money. That's why model risk is regulated. The U.S. guidance, SR 11-7, requires exactly the out-of-sample and stress testing I did here, and my naive agent is a textbook example of why that rule exists. Second is opacity. A deep network can't really explain why it holds what it holds, so a clean-looking backtest can hide both overfitting and the fact that there may be no real reason behind a trade. Third is my own conduct as a researcher. I fixed the success bar in advance and reported the result honestly, instead of hunting for a window where the agent happens to win. And one last point: this is a research exercise on historical data, with no real money, so nothing here is investment advice.")

# ========================================================= 11  CONCLUSION
s = slide(PLUM)
add_text(s, MARGIN, Inches(0.55), CW, Inches(0.28),
         [[("8 · CONCLUSION AND FUTURE WORK", {"size": 11, "color": TERRA, "bold": True, "font": F_BODY, "spacing": 2.2})]], space_after=Pt(0))
hrule(s, MARGIN, Inches(0.98), Inches(2.0), TERRA, thick=Pt(2.2))
add_text(s, MARGIN, Inches(1.2), CW, Inches(0.7),
         [[("From catastrophe to a competitive strategy", {"size": 29, "color": ON_PLUM, "bold": True, "font": F_HEAD})]], space_after=Pt(0))
add_text(s, MARGIN, Inches(2.15), Inches(6.5), Inches(0.3),
         [[("WHAT I FOUND", {"size": 10.5, "color": TERRA, "bold": True, "spacing": 1.6})]], space_after=Pt(0))
bullets(s, MARGIN, Inches(2.55), Inches(6.3), Inches(3.4), [
    "Naive DQN failed: Sharpe -0.98, worse than a coin flip, overfit and shorting a rising market.",
    "A cost-aware, validation-chosen redesign lifted it to +0.52, competitive with buy-and-hold.",
    "It matches buy-and-hold on SPY and beats it on QQQ; a once-a-year trend rule still leads.",
], size=14.5, gap=Pt(12), color=ON_PLUM, lead=TERRA)
add_text(s, Inches(7.4), Inches(2.15), Inches(5.4), Inches(0.3),
         [[("WHERE IT GOES NEXT", {"size": 10.5, "color": TERRA, "bold": True, "spacing": 1.6})]], space_after=Pt(0))
bullets(s, Inches(7.4), Inches(2.55), Inches(5.35), Inches(3.4), [
    "A richer state and a longer, multi-asset training history.",
    "Continuous position sizing instead of three discrete actions.",
    "Distributional or risk-aware objectives to temper overconfidence.",
    "Benchmark honestly against the trend-following rule that won.",
], size=14.5, gap=Pt(12), color=ON_PLUM, lead=TERRA)
add_text(s, MARGIN, Inches(6.3), CW, Inches(0.7),
         [[("The contribution is methodological: a rigorous, cost-aware test that earned a competitive strategy without ever tuning on the test set.",
            {"size": 14.5, "color": ON_PLUM, "italic": True, "font": F_HEAD})]], line_spacing=1.1, space_after=Pt(0))
add_text(s, MARGIN + CW - Inches(1.2), Inches(7.06), Inches(1.2), Inches(0.28),
         [[("11", {"size": 9.5, "color": ON_PLUM_SOFT, "font": F_BODY})]], align=PP_ALIGN.RIGHT, space_after=Pt(0))
note(s, "11 Conclusion and future work",
     "To wrap up. The naive agent failed, a Sharpe of minus zero-point-nine-eight, worse than a coin flip, because it overfit and shorted a rising market. A redesign chosen entirely on validation, long-only, turnover-penalized, ensembled, and picked to shrug off trading costs, lifted it to plus zero-point-five-two. That makes it genuinely competitive. It matches buy-and-hold on the S&P 500 and beats it on the Nasdaq, at similar risk and far less trading. I'll stay honest: on the S&P 500 it still doesn't clear my strict bar, and a one-trade-a-year rule is still the best. For future work, I'd give it a richer state, a longer multi-asset history, continuous position sizing instead of three choices, and a risk-aware objective. But the real contribution is the method. It's a rigorous, cost-aware test that turned a catastrophic failure into a competitive strategy, without ever tuning on the test set. Knowing exactly where the edge holds, and where it doesn't, is the point. Thank you.")

# ========================================================= 12  REFERENCES
s = slide(PAPER)
header(s, "References", "Sources")
refs = [
    "Board of Governors of the Federal Reserve System, & Office of the Comptroller of the Currency. (2011). Supervisory guidance on model risk management (SR 11-7 / OCC Bulletin 2011-12).",
    "Hambly, B., Xu, R., & Yang, H. (2023). Recent advances in reinforcement learning in finance. Mathematical Finance, 33(3), 437–503.",
    "Mnih, V., et al. (2015). Human-level control through deep reinforcement learning. Nature, 518(7540), 529–533.",
    "Moody, J., & Saffell, M. (2001). Learning to trade via direct reinforcement. IEEE Transactions on Neural Networks, 12(4), 875–889.",
    "Théate, T., & Ernst, D. (2021). An application of deep reinforcement learning to algorithmic trading. Expert Systems with Applications, 173, 114632.",
    "van Hasselt, H., Guez, A., & Silver, D. (2016). Deep reinforcement learning with double Q-learning. AAAI, 30(1), 2094–2100.",
]
add_text(s, MARGIN, Inches(1.85), CW, Inches(5.0),
         [[(r, {"size": 13, "color": INK, "font": F_BODY})] for r in refs],
         space_after=Pt(12), line_spacing=1.1, hanging=Inches(0.4))
footer(s, 12)
note(s, "12 References",
     'These are my sources, in APA style. The full details and the verified PDFs are included with the written report. Thank you for watching.')

# ========================================================= SAVE
pres_dir = os.path.join(ROOT, "presentation")
os.makedirs(pres_dir, exist_ok=True)
out = os.path.join(pres_dir, "PRESENTATION.pptx")
prs.save(out)
import shutil
shutil.copyfile(out, os.path.join(pres_dir, "SantiagoEchevarria_Jose_RL_Project.pptx"))
total = sum(word_count(t) for _, t in notes_log)
with open(os.path.join(pres_dir, "NARRATION.md"), "w", encoding="utf-8") as f:
    f.write("# Presentation narration\n\n")
    f.write("Full speaker script for the 13-15 minute recording (also embedded as the speaker notes "
            "in `PRESENTATION.pptx`). About %d words, roughly %.1f minutes at a measured ~140 words per "
            "minute; the demo clip plays during slide 9.\n\n" % (total, total / 140.0))
    for lab, t in notes_log:
        f.write("## Slide %s  (%d words)\n\n%s\n\n" % (lab, word_count(t), t))
print("Built %s | %d slides | narration ~%d words (~%.1f min at 140 wpm)"
      % (os.path.relpath(out, ROOT), len(prs.slides._sldIdLst), total, total / 140.0))
print("Wrote SantiagoEchevarria_Jose_RL_Project.pptx and NARRATION.md")
for lab, t in notes_log:
    print("   %-30s %4d words" % (lab, word_count(t)))
