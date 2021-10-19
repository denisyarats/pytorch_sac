import datetime
import io
import random
import traceback
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn


class ReplayBuffer:
    def __init__(self, specs, max_size, batch_size, nstep, discount):
        self._specs = {spec.name: spec for spec in specs}
        self._max_size = max_size
        self._batch_size = batch_size
        self._nstep = nstep
        self._discount = discount
        self._idx = 0
        self._full = False
        self._items = dict()
        for spec in specs:
            self._items[spec.name] = np.empty((max_size, *spec.shape),
                                              dtype=spec.dtype)

    def __len__(self):
        return self._max_size if self._full else self._idx

    def add(self, time_step):
        for spec in self._specs.values():
            value = time_step[spec.name]
            if np.isscalar(value):
                value = np.full(spec.shape, value, spec.dtype)
            assert spec.shape == value.shape and spec.dtype == value.dtype
            np.copyto(self._items[spec.name][self._idx], value)

        self._idx = (self._idx + 1) % self._max_size
        self._full = self._full or self._idx == 0

    def _sample(self):
        assert self._nstep <= len(self)
        idxs = np.random.randint(0,
                                 len(self) - self._nstep + 1,
                                 size=self._batch_size)

        obs = self._items['observation'][idxs]
        action = self._items['action'][idxs]
        next_obs = self._items['next_observation'][idxs + self._nstep - 1]

        reward = np.zeros((self._batch_size, *self._specs['reward'].shape),
                          dtype=self._specs['reward'].dtype)
        discount = np.ones((self._batch_size, *self._specs['discount'].shape),
                           dtype=self._specs['discount'].dtype)

        for i in range(self._nstep):
            reward = reward + discount * self._items['reward'][idxs + i]
            discount = discount * self._items['discount'][idxs +
                                                          i] * self._discount

        batch = (obs, action, reward, discount, next_obs)
        return batch

    def __iter__(self):
        while True:
            yield self._sample()
