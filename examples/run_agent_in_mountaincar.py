import tqdm
import gym
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from copy import deepcopy

import sys
sys.path.append('..')
from agents.dqn_agent import DQNAgent
# from agents.ddqn_agent import DDQNAgent
from agents.policy import EpsGreedyQPolicy
from agents.memory import RandomMemory
# from memory import SequentialMemory


def obs_processor(raw_obs):
    """convert raw observation to agent's observation"""
    return raw_obs


def build_q_network(input_shape, nb_output):
    input_layer = tf.keras.layers.Input(shape=input_shape)
    x = tf.keras.layers.Flatten()(input_layer)
    x = tf.keras.layers.Dense(32, activation='relu')(x)
    x = tf.keras.layers.Dense(32, activation='relu')(x)
    x = tf.keras.layers.Dense(32, activation='relu')(x)
    output_layer = tf.keras.layers.Dense(nb_output, activation='linear')(x)
    model = tf.keras.models.Model(inputs=input_layer, outputs=output_layer)

    return model


env = gym.make('MountainCar-v0')
np.random.seed(123)
env.seed(123)
nb_actions = env.action_space.n
actions = np.arange(nb_actions)
policy = EpsGreedyQPolicy(eps=1., eps_decay_rate=.9999, min_eps=.01)
memory = RandomMemory(limit=10000)
# memory = SequentialMemory(limit=50000, maxlen=1)
ini_observation = env.reset()
loss_fn = tf.keras.losses.Huber()
optimizer = tf.keras.optimizers.Adam()
model = build_q_network(
    input_shape=[
        len(ini_observation)],
    nb_output=len(actions))
target_model = build_q_network(
    input_shape=[
        len(ini_observation)],
    nb_output=len(actions))

agent = DQNAgent(actions=actions,
                 memory=memory,
                 update_interval=200,
                 train_interval=1,
                 batch_size=32,
                 observation=ini_observation,
                 model=model,
                 target_model=target_model,
                 policy=policy,
                 loss_fn=loss_fn,
                 optimizer=optimizer,
                 obs_processor=obs_processor, 
                 is_ddqn=True)

step_history = []
reward_history = []
nb_epsiodes = 1000
# for episode in range(nb_epsiodes):episode_reward_averague
episode_reward_average = -1
with tqdm.trange(nb_epsiodes) as t:
    for episode in t:
        # agent.reset()
        observation = env.reset()
        observation = deepcopy(observation)
        agent.observe(observation)
        done = False
        episode_reward = []
        step = 0
        # train
        while not done:
            action = agent.act()
            observation, reward, done, info = env.step(action)
            observation = deepcopy(observation)
            # reward = observation[0]
            if done:
                if step < 199:
                    reward = 100
                agent.observe(observation, reward, done)
                episode_reward.append(reward)

                episode_reward_average = 0.01*np.mean(episode_reward) + 0.99*episode_reward_average
                reward_history.append(np.mean(episode_reward))
                t.set_description('Episode {}, steps:{}, reward:{} '.format(episode, step, np.mean(episode_reward)))
                t.set_postfix(episode_reward=episode_reward_average)
                step_history.append(step)
                break
            else:
                agent.observe(observation, reward, done)
                episode_reward.append(reward)

            step += 1


        agent.training = True

env.close()
x = np.arange(len(step_history))
plt.ylabel('step')
plt.xlabel('episode')
plt.plot(x, step_history)
plt.savefig('result.png')
