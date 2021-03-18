
import gym
import numpy as np
import tqdm
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

def build_critic_network(input_state_shape,input_action_shape, nb_output):
    input_layer = tf.keras.layers.Input(shape=input_state_shape)
    x = tf.keras.layers.Flatten()(input_layer)
    x = tf.keras.layers.Dense(32, activation='relu')(x)
    x = tf.keras.layers.Dense(32, activation='relu')(x)

    input_action_layer = tf.keras.layers.Input(shape=input_action_shape)
    action_x = tf.keras.layers.Dense(32, activation='relu')(input_action_layer)
    x = tf.concat([x, action_x], axis=1)

    x = tf.keras.layers.Dense(32, activation='relu')(x)
    output_layer = tf.keras.layers.Dense(nb_output, activation='linear')(x)
    model = tf.keras.models.Model(inputs=[input_layer, input_action_layer], outputs=output_layer)

    return model

def build_actor_network(input_shape, nb_output):
    input_layer = tf.keras.layers.Input(shape=input_shape)
    x = tf.keras.layers.Flatten()(input_layer)
    x = tf.keras.layers.Dense(32, activation='relu')(x)
    x = tf.keras.layers.Dense(32, activation='relu')(x)
    x = tf.keras.layers.Dense(32, activation='relu')(x)
    output_layer = tf.keras.layers.Dense(nb_output, activation='linear')(x)
    model = tf.keras.models.Model(inputs=input_layer, outputs=output_layer)

    return model


env = gym.make('Pendulum-v0')
np.random.seed(123)
env.seed(123)
# nb_actions = env.action_space.n
#@todo action size
actions = np.arange(1)
policy = EpsGreedyQPolicy(eps=1., eps_decay_rate=.999, min_eps=.01)
memory = RandomMemory(limit=50000)
# memory = SequentialMemory(limit=50000, maxlen=1)
ini_observation = env.reset()
loss_fn = tf.keras.losses.Huber()
optimizer = tf.keras.optimizers.Adam()

critic = build_critic_network(
    input_state_shape=[
        len(ini_observation)],
    input_action_shape=[
        1],
    nb_output=1)
target_critic = build_critic_network(
    input_state_shape=[
        len(ini_observation)],
    input_action_shape=[
        1],
    nb_output=1)

actor = build_actor_network(
    input_shape=[
        len(ini_observation)],
    nb_output=1)
target_actor = build_actor_network(
    input_shape=[
        len(ini_observation)],
    nb_output=1)
agent = DDQNAgent(actions=actions,
                 memory=memory,
                 update_interval=200,
                 train_interval=1,
                 batch_size=32,
                 observation=ini_observation,
                 actor=actor,
                 target_actor=target_actor,
                 critic=critic,
                 target_critic=target_critic,
                 policy=policy,
                 loss_fn=loss_fn,
                 optimizer=optimizer,
                 obs_processor=obs_processor)

step_history = []
nb_epsiodes = 1000


episode_reward_average = -1
with tqdm.trange(nb_epsiodes) as t:
    for episode in t:
        done = False
        env.reset()
        while not done:
            action = 0
            observation, reward, done, info = env.step([action])
