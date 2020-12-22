
import gym
import numpy as np
import tqdm

env = gym.make('Pendulum-v0')
np.random.seed(123)
env.seed(123)

nb_epsiodes = 10
episode_reward_average = -1
with tqdm.trange(nb_epsiodes) as t:
    for episode in t:
        done = False
        env.reset()
        while not done:
            action = 0
            observation, reward, done, info = env.step([action])
