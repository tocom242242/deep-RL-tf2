import sys #nopep
sys.path.append('..')#nopep

import gym
import numpy as np
import tqdm
import matplotlib.pyplot as plt
import tensorflow as tf
from copy import deepcopy

from agents.ddpg_agent import DDPGAgent
import agents.policy as policy
import agents.memorys as memorys


def obs_processor(raw_obs):
    """convert raw observation to agent's observation"""
    return raw_obs


def build_critic_network(input_state_shape, input_action_shape, nb_output):
    input_state_layer = tf.keras.layers.Input(shape=input_state_shape)
    state_hidden_layer = tf.keras.layers.Flatten()(input_state_layer)
    state_hidden_layer = tf.keras.layers.Dense(
        32, activation='relu')(state_hidden_layer)
    state_hidden_layer = tf.keras.layers.Dense(
        32, activation='relu')(state_hidden_layer)

    input_action_layer = tf.keras.layers.Input(shape=input_action_shape)
    action_hidden_layer = tf.keras.layers.Dense(
        32, activation='relu')(input_action_layer)

    x = tf.concat([state_hidden_layer, action_hidden_layer], axis=1)

    x = tf.keras.layers.Dense(256, activation='relu')(x)
    x = tf.keras.layers.Dense(256, activation='relu')(x)
    output_layer = tf.keras.layers.Dense(nb_output, activation='linear')(x)
    model = tf.keras.models.Model(
        inputs=[
            input_state_layer,
            input_action_layer],
        outputs=output_layer)

    return model


def build_actor_network(input_shape, nb_output):
    input_layer = tf.keras.layers.Input(shape=input_shape)
    x = tf.keras.layers.Flatten()(input_layer)
    x = tf.keras.layers.Dense(256, activation='relu')(x)
    x = tf.keras.layers.Dense(256, activation='relu')(x)
    output_layer = tf.keras.layers.Dense(nb_output, activation='linear')(x)
    model = tf.keras.models.Model(inputs=input_layer, outputs=output_layer)

    return model


env = gym.make('Pendulum-v0')
np.random.seed(123)
env.seed(123)

memory = memorys.RandomMemory(limit=50000)
# memory = SequentialMemory(limit=50000, maxlen=1)
ini_observation = env.reset()
loss_fn = tf.keras.losses.mse
optimizer = tf.keras.optimizers.Adam(lr=0.002)

step_history = []
nb_epsiodes = 100

reward_history = []

critic = build_critic_network(
    input_state_shape=[
        len(ini_observation)],
    input_action_shape=[1],
    nb_output=1)
target_critic = build_critic_network(
    input_state_shape=[
        len(ini_observation)],
    input_action_shape=[1],
    nb_output=1)

actor = build_actor_network(
    input_shape=[len(ini_observation)],
    nb_output=1)
target_actor = build_actor_network(
    input_shape=[len(ini_observation)],
    nb_output=1)

action_noise = policy.OUActionNoise(
    mean=np.zeros(1),
    std_deviation=float(0.2) *
    np.ones(1))

agent = DDPGAgent(
    action_noise=action_noise,
    memory=memory,
    update_interval=0.005,
    train_interval=1,
    batch_size=32,
    observation=ini_observation,
    actor=actor,
    target_actor=target_actor,
    critic=critic,
    target_critic=target_critic,
    loss_fn=loss_fn,
    optimizer=optimizer,
    obs_processor=obs_processor)

with tqdm.trange(nb_epsiodes) as t:
    for episode in t:
        observation = env.reset()
        observation = deepcopy(observation)
        agent.observe(observation)
        done = False
        step = 0
        episode_reward_history = []
        while not done:
            action = agent.act()
            observation, reward, done, info = env.step([action])
            step += 1
            observation = deepcopy(observation)
            episode_reward_history.append(reward)
            agent.observe(observation, reward, done)
            if done:
                t.set_description(
                    'Episode {}, Reward:{}'.format(
                        episode, np.sum(episode_reward_history)))
                t.set_postfix(episode_reward=np.sum(episode_reward_history))
                step_history.append(step)
                break

        reward_history.append(np.sum(episode_reward_history))

env.close()
x = np.arange(len(reward_history))
plt.ylabel('episode reward')
plt.xlabel('episode')
plt.plot(x, reward_history)
plt.savefig('result.png')
plt.show()
