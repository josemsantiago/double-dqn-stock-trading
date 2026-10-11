"""The Double-DQN agent: a small Q-network, an experience replay buffer, and the update.

The Q-network is a multilayer perceptron that maps the state to one estimated value
per position (short, flat, long). Training follows the deep Q-network of Mnih et al.
(2015), with uniform experience replay and a target network that is copied from the
online network on a fixed schedule, and it uses the double-Q target of van Hasselt et
al. (2016): the online network picks the best next action and the target network
values it, which removes most of the upward bias a plain max over noisy estimates
builds into the target. The loss is the Huber loss, optimized by Adam with the gradient
norm clipped and an optional L2 weight penalty.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Network
# ---------------------------------------------------------------------------
class QNetwork(nn.Module):
    def __init__(self, n_inputs, n_actions, hidden=128):
        super().__init__()
        self.n_actions = n_actions
        self.net = nn.Sequential(
            nn.Linear(n_inputs, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, x):
        return self.net(x)


# ---------------------------------------------------------------------------
# Replay buffer
# ---------------------------------------------------------------------------
class ReplayBuffer:
    """A fixed-size ring buffer of transitions, sampled uniformly at random."""

    def __init__(self, capacity, obs_dim, rng):
        self.capacity, self.rng = capacity, rng
        self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.act = np.zeros(capacity, dtype=np.int64)
        self.rew = np.zeros(capacity, dtype=np.float32)
        self.done = np.zeros(capacity, dtype=np.float32)
        self.size = self.ptr = 0

    def add(self, obs, act, rew, next_obs, done):
        i = self.ptr
        self.obs[i], self.act[i], self.rew[i], self.next_obs[i], self.done[i] = obs, act, rew, next_obs, done
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        idx = self.rng.integers(0, self.size, size=batch_size)
        return (torch.from_numpy(self.obs[idx]), torch.from_numpy(self.act[idx]),
                torch.from_numpy(self.rew[idx]), torch.from_numpy(self.next_obs[idx]),
                torch.from_numpy(self.done[idx]))


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------
class DoubleDQN:
    def __init__(self, obs_dim, n_actions, lr, gamma, hidden=128, target_sync=1000, grad_clip=10.0,
                 weight_decay=0.0, device="cpu"):
        self.device = torch.device(device)
        self.n_actions = n_actions
        self.online = QNetwork(obs_dim, n_actions, hidden=hidden).to(self.device)
        self.target = QNetwork(obs_dim, n_actions, hidden=hidden).to(self.device)
        self.target.load_state_dict(self.online.state_dict())
        self.target.eval()
        self.opt = torch.optim.Adam(self.online.parameters(), lr=lr, weight_decay=weight_decay)
        self.gamma, self.target_sync, self.grad_clip = gamma, target_sync, grad_clip
        self.updates = 0

    @torch.no_grad()
    def q_values(self, obs):
        x = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        return self.online(x.unsqueeze(0) if x.dim() == 1 else x).cpu().numpy()

    def act(self, obs, epsilon, rng):
        """Epsilon-greedy action: random with probability epsilon, else the greedy one."""
        if rng.random() < epsilon:
            return int(rng.integers(0, self.n_actions))
        return int(np.argmax(self.q_values(obs)[0]))

    def update(self, batch):
        """One gradient step on the double-Q target. Returns (loss, mean Q of the batch)."""
        obs, act, rew, next_obs, done = (t.to(self.device) for t in batch)
        q = self.online(obs).gather(1, act.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            best_next = self.online(next_obs).argmax(dim=1, keepdim=True)       # online picks
            next_q = self.target(next_obs).gather(1, best_next).squeeze(1)      # target values
            target = rew + self.gamma * (1.0 - done) * next_q
        loss = F.smooth_l1_loss(q, target)
        self.opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.online.parameters(), self.grad_clip)
        self.opt.step()
        self.updates += 1
        if self.updates % self.target_sync == 0:
            self.target.load_state_dict(self.online.state_dict())
        return float(loss.item()), float(q.mean().item())
