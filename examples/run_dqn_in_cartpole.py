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
# from agents.memory import SequentialMemory


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


env = gym.make('CartPole-v0')
np.random.seed(123)
env.seed(123)
nb_actions = env.action_space.n
actions = np.arange(nb_actions)
policy = EpsGreedyQPolicy(eps=1., eps_decay_rate=.999, min_eps=.01)
memory = RandomMemory(limit=50000)
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
                 is_ddqn=False)

step_history = []
nb_epsiodes = 1000
with tqdm.trange(nb_epsiodes) as t:
    for episode in t:
        # agent.reset()
        observation = env.reset()
        observation = deepcopy(observation)
        agent.observe(observation)
        done = False
        step = 0
        episode_reward_history = []
        # train
        while not done:
            action = agent.act()
            observation, reward, done, info = env.step(action)
            step += 1
            observation = deepcopy(observation)
            episode_reward_history.append(reward)
            agent.observe(observation, reward, done)
            if done:
                t.set_description('Episode {}: {} steps'.format(episode, step))
                t.set_postfix(episode_reward=np.sum(episode_reward_history))
                step_history.append(step)
                break

        # if last step is bigger than 195, stop the game.
        if all(np.array(step_history[-10:]) >= (env.spec.max_episode_steps - 5)):
            print('Problem is solved in {} episodes.'.format(episode))
            break

        agent.training = True

env.close()
x = np.arange(len(step_history))
plt.ylabel('step')
plt.xlabel('episode')
plt.plot(x, step_history)
plt.savefig('result.png')
