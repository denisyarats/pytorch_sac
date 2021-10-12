import torch
import torch.nn as nn
from distribs import TruncatedNormal, SquashedNormal
from torch.nn.utils import spectral_norm
from utils import weight_init

        


class DoubleQCritic(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dim, use_ln, use_sn):
        super().__init__()
        
        act = lambda : nn.Sequential(nn.LayerNorm(hidden_dim), nn.Tanh()) if use_ln else nn.ReLU()
        sn = spectral_norm if use_sn else lambda m: m

        self.Q1 = nn.Sequential(nn.Linear(obs_dim + action_dim, hidden_dim),
                                act(),
                                sn(nn.Linear(hidden_dim, hidden_dim)), nn.ReLU(),
                                nn.Linear(hidden_dim, 1))

        self.Q2 = nn.Sequential(nn.Linear(obs_dim + action_dim, hidden_dim),
                                act(),
                                sn(nn.Linear(hidden_dim, hidden_dim)), nn.ReLU(),
                                nn.Linear(hidden_dim, 1))

        self.apply(weight_init)

    def forward(self, obs, action):
        obs_action = torch.cat([obs, action], dim=-1)
        q1 = self.Q1(obs_action)
        q2 = self.Q2(obs_action)

        return q1, q2


class DeterministicActor(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dim, use_ln, use_sn):
        super().__init__()
        
        act = lambda : nn.Sequential(nn.LayerNorm(hidden_dim), nn.Tanh()) if use_ln else nn.ReLU()
        sn = spectral_norm if use_sn else lambda m: m

        self.policy = nn.Sequential(nn.Linear(obs_dim, hidden_dim),
                                    act(),
                                    sn(nn.Linear(hidden_dim,
                                              hidden_dim)), nn.ReLU(),
                                    nn.Linear(hidden_dim, action_dim))

        self.apply(weight_init)

    def forward(self, obs, std):
        mu = self.policy(obs)
        mu = torch.tanh(mu)
        std = torch.ones_like(mu) * std

        dist = TruncatedNormal(mu, std)
        return dist


class StochasticActor(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dim, log_std_bounds):
        super().__init__()

        self.log_std_bounds = log_std_bounds
        self.policy = nn.Sequential(nn.Linear(obs_dim, hidden_dim),
                                    nn.LayerNorm(hidden_dim), nn.Tanh(),
                                    nn.Linear(hidden_dim, hidden_dim),
                                    nn.ReLU(),
                                    nn.Linear(hidden_dim, 2 * action_dim))

        self.apply(weight_init)

    def forward(self, obs):
        mu, log_std = self.policy(obs).chunk(2, dim=-1)

        # constrain log_std inside [log_std_min, log_std_max]
        log_std = torch.tanh(log_std)
        log_std_min, log_std_max = self.log_std_bounds
        log_std = log_std_min + 0.5 * (log_std_max - log_std_min) * (log_std +
                                                                     1)

        std = log_std.exp()

        dist = SquashedNormal(mu, std)
        return dist
