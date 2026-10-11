"""One training run of the Double-DQN trading agent.

Training draws one-year episodes (252 trading days) from random start dates inside the
2010-2018 training block, acting epsilon-greedily with epsilon falling linearly from 1.0
to 0.05 over the first 25,000 steps. After 2,000 warm-up steps the agent takes one
gradient step per environment step on a minibatch of 64 transitions from a 50,000-step
replay buffer, and the target network is refreshed every 1,000 updates.

Every ten episodes the greedy policy is run once through the 2019-2020 validation block
and scored by its annualized Sharpe ratio after costs. Once exploration has finished
(epsilon at its floor, from episode 100 on), the checkpoint with the best validation
Sharpe is the one kept, so the validation block chooses when to stop and the test block
is never touched here. Earlier checkpoints are not eligible: a barely trained network
tends to output one action everywhere, and on a rising validation window that can
score well without the agent having learned anything. The run also logs the training reward, loss, and
mean Q-value per episode, and how much the greedy validation policy changes between
evaluations, which shows when the learned policy settles.
"""

import argparse
import copy
import json
import os
import time

import numpy as np
import torch

import data
import evaluate
from agent import DoubleDQN, ReplayBuffer
from env import TradingEnv

SETTINGS = {
    "episodes": 200,
    "episode_len": 252,
    "buffer": 50_000,
    "batch": 64,
    "learn_start": 2_000,
    "eps_start": 1.0,
    "eps_end": 0.05,
    "eps_decay_steps": 25_000,
    "target_sync": 1_000,
    "hidden": 64,            # regularized network (see the regularization stage)
    "weight_decay": 1e-3,    # L2 weight penalty
    "obs_noise": 0.5,        # Gaussian noise on standardized training observations
    "long_only": False,      # action set: short/flat/long (True = flat/long only)
    "turnover_penalty": 0.0, # extra training-only reward penalty per unit traded
    "eval_every": 10,
    "select_from": 100,      # checkpoints count only once epsilon has reached its floor
    "cost": 0.0005,          # 5 basis points per unit of position traded
}


def epsilon_at(step, s):
    frac = min(1.0, step / s["eps_decay_steps"])
    return s["eps_start"] + frac * (s["eps_end"] - s["eps_start"])


def train_run(ticker, config, seed, device="cpu", settings=None, verbose=False):
    """Train one agent and return its best-validation checkpoint and the run's logs.

    config holds the tuned settings: reward ('logret' or 'dsr'), gamma, lr, window.
    """
    s = dict(SETTINGS, **(settings or {}))
    last_eval = (s["episodes"] // s["eval_every"]) * s["eval_every"]
    if last_eval < s["select_from"]:
        raise ValueError("episodes=%d never reaches an evaluation at or after select_from=%d "
                         "(eval_every=%d), so no checkpoint would be kept" % (s["episodes"], s["select_from"], s["eval_every"]))
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    blocks, norm = data.make_blocks(ticker, config["window"])

    env = TradingEnv(blocks["train"], cost=s["cost"], reward=config["reward"], episode_len=s["episode_len"],
                     obs_noise=s["obs_noise"], long_only=config.get("long_only", s["long_only"]),
                     turnover_penalty=config.get("turnover_penalty", s["turnover_penalty"]))
    obs_dim = env.observation_space.shape[0]
    n_actions = env.n_actions
    agent = DoubleDQN(obs_dim, n_actions, lr=config["lr"], gamma=config["gamma"], hidden=s["hidden"],
                      target_sync=s["target_sync"], weight_decay=s["weight_decay"], device=device)
    buffer = ReplayBuffer(s["buffer"], obs_dim, rng)

    logs = {"episode_reward": [], "loss": [], "mean_q": [], "epsilon": [],
            "eval_episode": [], "eval_steps": [], "val_sharpe": [], "val_policy_change": []}
    best = {"val_sharpe": -np.inf, "episode": 0, "state": None}
    prev_positions, steps, t0 = None, 0, time.time()

    for ep in range(s["episodes"]):
        obs, _ = env.reset(seed=int(rng.integers(0, 2**31 - 1)))
        ep_reward, losses, qs, done = 0.0, [], [], False
        while not done:
            eps = epsilon_at(steps, s)
            action = agent.act(obs, eps, rng)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            buffer.add(obs, action, reward, next_obs, float(terminated))
            obs, ep_reward, steps = next_obs, ep_reward + reward, steps + 1
            done = terminated or truncated
            if steps >= s["learn_start"]:
                loss, q = agent.update(buffer.sample(s["batch"]))
                losses.append(loss)
                qs.append(q)
        logs["episode_reward"].append(ep_reward)
        logs["loss"].append(float(np.mean(losses)) if losses else float("nan"))
        logs["mean_q"].append(float(np.mean(qs)) if qs else float("nan"))
        logs["epsilon"].append(eps)

        if (ep + 1) % s["eval_every"] == 0:
            run = evaluate.run_policy(blocks["val"], evaluate.greedy(agent), s["cost"],
                                      long_only=config.get("long_only", False))
            sharpe = evaluate.metrics(run)["sharpe"]
            change = float(np.mean(run["positions"] != prev_positions)) if prev_positions is not None else float("nan")
            prev_positions = run["positions"]
            logs["eval_episode"].append(ep + 1)
            logs["eval_steps"].append(steps)
            logs["val_sharpe"].append(sharpe)
            logs["val_policy_change"].append(change)
            if ep + 1 >= s["select_from"] and sharpe > best["val_sharpe"]:
                best = {"val_sharpe": sharpe, "episode": ep + 1, "state": copy.deepcopy(agent.online.state_dict())}
            if verbose:
                print("ep %3d  steps %6d  eps %.2f  reward %7.2f  loss %.4f  Q %6.3f  val Sharpe %5.2f  change %.2f  (%.0fs)"
                      % (ep + 1, steps, eps, ep_reward, logs["loss"][-1], logs["mean_q"][-1], sharpe,
                         change if change == change else 0.0, time.time() - t0))

    return {"ticker": ticker, "config": config, "seed": seed, "obs_dim": obs_dim, "n_actions": n_actions, "settings": s,
            "best_val_sharpe": float(best["val_sharpe"]), "best_episode": best["episode"],
            "state": best["state"], "norm": norm, "logs": logs, "seconds": time.time() - t0}


def load_agent(result_or_path, device="cpu"):
    """Rebuild a greedy agent from a train_run result or a saved checkpoint file."""
    if isinstance(result_or_path, str):
        ck = torch.load(result_or_path, map_location=device, weights_only=True)
    else:
        ck = result_or_path
    agent = DoubleDQN(ck["obs_dim"], ck.get("n_actions", 3), lr=ck["config"]["lr"], gamma=ck["config"]["gamma"],
                      hidden=ck["settings"]["hidden"], device=device)
    agent.online.load_state_dict(ck["state"])
    agent.online.eval()
    return agent


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Train one Double-DQN trading agent.")
    ap.add_argument("--ticker", default="SPY")
    ap.add_argument("--reward", default="logret", choices=["logret", "dsr"])
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--window", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--episodes", type=int, default=SETTINGS["episodes"])
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    torch.set_num_threads(1)
    cfg = {"reward": args.reward, "gamma": args.gamma, "lr": args.lr, "window": args.window}
    res = train_run(args.ticker, cfg, args.seed, device=args.device,
                    settings={"episodes": args.episodes}, verbose=True)
    print(json.dumps({k: res[k] for k in ("best_val_sharpe", "best_episode", "seconds")}, indent=1))
