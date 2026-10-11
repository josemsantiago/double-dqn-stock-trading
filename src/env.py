"""A Gymnasium environment for trading one asset day by day.

At the close of each trading day the agent sees a window of recent returns, a few
technical features, and the position it currently holds, and it chooses its position for
the next day. The position is held from that close to the next one, so the step earns
the next day's return times the position, less a proportional cost on any change of
position. A flip from short to long trades two units and pays twice.

The action set is either short/flat/long or, when `long_only` is set, just flat/long.
Long-only removes the structural drag of shorting an index that drifts upward, and it is
the setting the improved agent uses.

Two rewards are available. The default is the change in the log value of the account,
scaled by 100 so a 1% day is a reward near 1 (a positive rescaling leaves the optimal
policy unchanged and keeps the Huber loss in its quadratic range). The alternative is
the differential Sharpe ratio of Moody and Saffell (2001), which rewards a step by how
much it raises an exponentially weighted Sharpe ratio of the account's returns. An
optional turnover penalty is subtracted from the reward (not from the real returns) to
discourage churn during training; the evaluation metrics always use the real returns.

Episodes never terminate in the Markov sense: a trading account has no absorbing state,
so an episode only ends by truncation (the end of a block or the episode-length limit),
and the agent bootstraps through every boundary.
"""

import gymnasium as gym
import numpy as np
from gymnasium import spaces

POSITIONS = np.array([-1.0, 0.0, 1.0])   # short, flat, long (the full action set)
LONG_ONLY_POSITIONS = np.array([0.0, 1.0])
ACTION_NAMES = {3: ["short", "flat", "long"], 2: ["flat", "long"]}


class TradingEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, block, cost=0.0005, reward="logret", episode_len=None,
                 dsr_eta=0.01, dsr_init_vol=0.01, obs_noise=0.0,
                 long_only=False, turnover_penalty=0.0):
        """block: one entry from data.make_blocks. cost: proportional cost per unit of
        position traded (0.0005 is 5 basis points). episode_len: random-start episodes of
        this many steps for training; None runs the whole block from its start. obs_noise:
        Gaussian noise on the standardized features, used only in training. long_only:
        restrict the action set to flat and long. turnover_penalty: an extra reward penalty
        per unit traded, applied in training only to discourage churn."""
        super().__init__()
        self.x = block["x"]
        self.next_ret = block["next_ret"]
        self.dates = block["dates"]
        self.n_steps = len(self.next_ret) - 1          # the terminal row has no decision
        self.cost = float(cost)
        self.reward_type = reward
        self.episode_len = episode_len
        self.dsr_eta = dsr_eta
        self.dsr_init_var = dsr_init_vol ** 2
        self.obs_noise = float(obs_noise)
        self.turnover_penalty = float(turnover_penalty)
        self.positions = LONG_ONLY_POSITIONS if long_only else POSITIONS
        self.n_actions = len(self.positions)
        self.flat_idx = int(np.where(self.positions == 0.0)[0][0])
        n_feat = self.x.shape[1] + self.n_actions
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(n_feat,), dtype=np.float32)
        self.action_space = spaces.Discrete(self.n_actions)

    # -----------------------------------------------------------------------
    def _obs(self):
        onehot = np.zeros(self.n_actions, dtype=np.float32)
        onehot[self.pos_idx] = 1.0
        feats = self.x[self.t]
        if self.obs_noise > 0.0:
            feats = feats + self.np_random.normal(0.0, self.obs_noise, size=feats.shape).astype(np.float32)
        return np.concatenate([feats, onehot])

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        options = options or {}
        if "start" in options:
            self.start = int(options["start"])
        elif self.episode_len is not None and self.n_steps > self.episode_len:
            self.start = int(self.np_random.integers(0, self.n_steps - self.episode_len + 1))
        else:
            self.start = 0
        self.end = self.n_steps if self.episode_len is None else min(self.n_steps, self.start + self.episode_len)
        self.t = self.start
        self.pos_idx = self.flat_idx                       # every episode starts flat
        self.dsr_a, self.dsr_b = 0.0, self.dsr_init_var
        return self._obs(), {"date": self.dates[self.t]}

    def step(self, action):
        new_pos = self.positions[int(action)]
        traded = abs(new_pos - self.positions[self.pos_idx])
        cost = self.cost * traded
        net = new_pos * self.next_ret[self.t] - cost        # simple return of the account

        if self.reward_type == "logret":
            reward = 100.0 * np.log1p(net)
        elif self.reward_type == "dsr":
            # Differential Sharpe ratio (Moody & Saffell, 2001): the first-order effect of
            # this return on an exponential moving Sharpe ratio, with A and B the moving
            # first and second moments. B starts at a 1%-daily-volatility prior so the
            # first steps are not divided by zero.
            a, b = self.dsr_a, self.dsr_b
            da, db = net - a, net * net - b
            denom = max(b - a * a, 1e-12) ** 1.5
            reward = float(np.clip((b * da - 0.5 * a * db) / denom, -10.0, 10.0))
            self.dsr_a, self.dsr_b = a + self.dsr_eta * da, b + self.dsr_eta * db
        else:
            raise ValueError("unknown reward %r" % self.reward_type)
        reward -= self.turnover_penalty * traded

        info = {"date": self.dates[self.t], "position": new_pos, "net_return": net,
                "cost": cost, "traded": traded}
        self.pos_idx = int(action)
        self.t += 1
        truncated = self.t >= self.end
        return self._obs(), float(reward), False, truncated, info
