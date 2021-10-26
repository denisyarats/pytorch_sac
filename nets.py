import torch
from torch import nn, jit
from distribs import TruncatedNormal, SquashedNormal
from torch.nn.utils import spectral_norm
from utils import weight_init


def maybe_sn(m, use_sn):
    return spectral_norm(m) if use_sn else m


class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dims, spectral_norms):
        super().__init__()
        assert len(hidden_dims) == len(spectral_norms)
        layers = []
        for dim, use_sn in zip(hidden_dims, spectral_norms):
            layers += [
                maybe_sn(nn.Linear(input_dim, dim), use_sn),
                nn.ReLU(inplace=True)
            ]
            input_dim = dim

        layers += [nn.Linear(input_dim, output_dim)]

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class LayerNormMLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dims, spectral_norms):
        super().__init__()
        assert len(hidden_dims) == len(spectral_norms)

        self.net = nn.Sequential(
            maybe_sn(nn.Linear(input_dim, hidden_dims[0]), spectral_norms[0]),
            nn.LayerNorm(hidden_dims[0]), nn.Tanh(),
            MLP(hidden_dims[0], output_dim, hidden_dims[1:],
                spectral_norms[1:]))

    def forward(self, x):
        return self.net(x)


class DoubleQCritic(nn.Module):
    def __init__(self, obs_dim, action_dim, use_ln, hidden_dims,
                 spectral_norms):
        super().__init__()

        input_dim = obs_dim + action_dim
        mlp_type = LayerNormMLP if use_ln else MLP

        self.q1_net = mlp_type(input_dim, 1, hidden_dims, spectral_norms)
        self.q2_net = mlp_type(input_dim, 1, hidden_dims, spectral_norms)

        self.apply(weight_init)

    def forward(self, obs, action):
        obs_action = torch.cat([obs, action], dim=-1)
        q1 = self.q1_net(obs_action)
        q2 = self.q2_net(obs_action)

        return q1, q2


class DeterministicActor(nn.Module):
    def __init__(self, obs_dim, action_dim, use_ln, hidden_dims,
                 spectral_norms):
        super().__init__()

        mlp_type = LayerNormMLP if use_ln else MLP
        self.policy_net = mlp_type(obs_dim, action_dim, hidden_dims,
                                   spectral_norms)

        self.apply(weight_init)

    def forward(self, obs, std):
        mu = self.policy_net(obs)
        mu = torch.tanh(mu)
        std = torch.ones_like(mu) * std

        dist = TruncatedNormal(mu, std)
        return dist


class StochasticActor(nn.Module):
    def __init__(self, obs_dim, action_dim, use_ln, hidden_dims,
                 spectral_norms, log_std_bounds):
        super().__init__()

        self.log_std_bounds = log_std_bounds

        mlp_type = LayerNormMLP if use_ln else MLP
        self.policy_net = mlp_type(obs_dim, 2 * action_dim, hidden_dims,
                                   spectral_norms)

        self.apply(weight_init)

    def forward(self, obs):
        mu, log_std = self.policy_net(obs).chunk(2, dim=-1)

        # constrain log_std inside [log_std_min, log_std_max]
        log_std = torch.tanh(log_std)
        log_std_min, log_std_max = self.log_std_bounds
        log_std = log_std_min + 0.5 * (log_std_max - log_std_min) * (log_std +
                                                                     1)

        std = log_std.exp()

        dist = SquashedNormal(mu, std)
        return dist
