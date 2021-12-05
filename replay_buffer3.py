import datetime
import io
import random
import traceback
from collections import defaultdict, deque

import numpy as np
import torch
import torch.nn as nn


class ReplayBuffer:
    def __init__(self, specs, max_size, batch_size, nstep, discount):
        self._specs = specs
        self._max_size = max_size
        self._batch_size = batch_size
        self._nstep = nstep
        self._discount = discount
        self._idx = 0
        self._full = False
        self._items = dict()
        self._queue = deque([], maxlen=nstep + 1)
        for spec in specs:
            self._items[spec.name] = np.empty((max_size, *spec.shape),
                                              dtype=spec.dtype)
    def __len__(self):
        return self._max_size if self._full else self._idx

    def add(self, time_step):
        for spec in self._specs:
            if spec.name == 'next_observation': continue
            value = time_step[spec.name]
            if np.isscalar(value):
                value = np.full(spec.shape, value, spec.dtype)
            assert spec.shape == value.shape and spec.dtype == value.dtype

        self._queue.append(time_step)
        if len(self._queue) == self._nstep + 1:
            np.copyto(self._items['observation'][self._idx],
                      self._queue[0].observation)
            np.copyto(self._items['action'][self._idx], self._queue[1].action)
            np.copyto(self._items['next_observation'][self._idx],
                      self._queue[-1].observation)
            reward, discount = 0.0, 1.0
            self._queue.popleft()
            for ts in self._queue:
                reward += discount * ts.reward
                discount *= ts.discount * self._discount
            np.copyto(self._items['reward'][self._idx], reward)
            np.copyto(self._items['discount'][self._idx], discount)

            self._idx = (self._idx + 1) % self._max_size
            self._full = self._full or self._idx == 0

        if time_step.last():
            self._queue.clear()

    def _sample(self):
        idxs = np.random.randint(0, len(self), size=self._batch_size)
        batch = tuple(self._items[spec.name][idxs] for spec in self._specs)
        return batch

    def __iter__(self):
        while True:
            yield self._sample()
