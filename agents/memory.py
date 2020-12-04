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
        self.experiences = deque(maxlen=limit)

    def sample(self, batch_size):
        assert batch_size > 1, "batch_size must be positive integer"

        batch_size = min(batch_size, len(self.experiences))
        mini_batch = random.sample(self.experiences, batch_size)
        state_batch = []
        action_batch = []
        reward_batch = []
        next_state_batch = []
        terminal_batch = []
        for state, action, reward, next_state, done in mini_batch:
            state_batch.append(state)
            action_batch.append(action)
            reward_batch.append(reward)
            next_state_batch.append(next_state)
            terminal_batch.append(0. if done else 1.)

        state_batch = np.array(state_batch)
        action_batch = np.array(action_batch)
        reward_batch = np.array(reward_batch)
        next_state_batch = np.array(next_state_batch)
        terminal_batch = np.array(terminal_batch)

        assert type(state_batch).__module__ == np.__name__
        assert type(action_batch).__module__ == np.__name__
        assert type(reward_batch).__module__ == np.__name__
        assert type(next_state_batch).__module__ == np.__name__
        assert type(terminal_batch).__module__ == np.__name__
        
        return state_batch, action_batch, reward_batch, next_state_batch, terminal_batch

    def append(self, state, action, reward, next_state, terminal=False):
        assert state is not None
        assert action is not None
        assert reward is not None
        assert next_state is not None
        assert terminal is not None

        self.experiences.append((state, action, reward, next_state, terminal))



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
