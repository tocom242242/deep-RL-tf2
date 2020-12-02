import random
import numpy as np
from collections import deque, namedtuple

Experience = namedtuple(
    'Experience', 'state0, action, reward, state1, terminal')


def sample_batch_indexes(low, high, size):
    r = range(low, high)
    batch_idxs = random.sample(r, size)

    return batch_idxs


class Memory:
    def __init__(self, limit, maxlen):
        self.actions = deque(maxlen=limit)
        self.rewards = deque(maxlen=limit)
        self.terminals = deque(maxlen=limit)
        self.observations = deque(maxlen=limit)
        self.maxlen = maxlen
        self.recent_observations = deque(maxlen=maxlen)

    def sample(self, batch_size):
        batch_idxs = sample_batch_indexes(
            0, len(self.observations) - 1, size=batch_size)
        for (i, idx) in enumerate(batch_idxs):
            terminal = self.terminals[idx-1]
            while terminal:
                idx = sample_batch_indexes(
                    0, len(self.observations)-1, size=1)[0]
                terminal = self.terminals[idx-1]
            batch_idxs[i] = idx

        experiences = []
        for idx in batch_idxs:
            state0 = self.observations[idx]
            action = self.actions[idx]
            reward = self.rewards[idx]
            terminal = self.terminals[idx]
            state1 = self.observations[idx+1]
            experiences.append(Experience(state0=state0,
                                          action=action,
                                          reward=reward,
                                          state1=state1,
                                          terminal=terminal))

        return experiences

    def append(self, observation, action, reward, terminal=False):
        self.observations.append(observation)
        self.actions.append(action)
        self.rewards.append(reward)
        self.terminals.append(terminal)
        self.recent_observations.append(observation)


class SequentialMemory:
    def __init__(self, limit, maxlen):
        self.actions = deque(maxlen=limit)
        self.rewards = deque(maxlen=limit)
        self.terminals = deque(maxlen=limit)
        self.observations = deque(maxlen=limit)
        self.maxlen = maxlen
        self.recent_observations = deque(maxlen=maxlen)

    def sample(self, batch_size):
        batch_idxs = []
        all_idxs = np.arange(0, len(self.observations)-1)
        while True:
            idx = np.random.choice(all_idxs)
            terminal = self.terminals[idx-1]
            while terminal is False and len(batch_idxs) < batch_size:
                batch_idxs.append(idx)
                # print(idx)
                # print(len(batch_idxs))
                all_idxs = all_idxs[all_idxs != idx]
                idx=idx+1
                if idx <= len(self.observations):
                    break
                terminal = self.terminals[idx-1]

            if len(batch_idxs) >= batch_size:
                break

        experiences = []
        for idx in batch_idxs:
            state0 = self.observations[idx]
            action = self.actions[idx]
            reward = self.rewards[idx]
            terminal = self.terminals[idx]
            state1 = self.observations[idx+1]
            experiences.append(Experience(state0=state0,
                                          action=action,
                                          reward=reward,
                                          state1=state1,
                                          terminal=terminal))

        return experiences

    def append(self, observation, action, reward, terminal=False):
        self.observations.append(observation)
        self.actions.append(action)
        self.rewards.append(reward)
        self.terminals.append(terminal)
        self.recent_observations.append(observation)
