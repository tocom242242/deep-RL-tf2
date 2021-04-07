import tqdm
import gym
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from copy import deepcopy

# import sys
# sys.path.append('..')
# from agents.drqn_agent import DRQNAgent
# from agents.policy import EpsGreedyQPolicy
# import agents.memorys as memorys

from drqn_agent import DRQNAgent
from policy import EpsGreedyQPolicy
import memorys as memorys


def obs_processor(raw_obs):
    """convert raw observation to agent's observation"""
    return raw_obs


time_step = 4

def build_recurrent_q_network(input_shape, nb_output):
    input_layer = tf.keras.layers.Input(shape=input_shape)
    # input_layer = tf.keras.layers.Input(batch_shape=[time_step,]+list(input_shape))
    # x = tf.keras.layers.Flatten()(input_layer)
    x = tf.keras.layers.Dense(32, activation='relu')(input_layer)
    # x = tf.keras.layers.LSTM(32, stateful=True)(x)
    # x,hidden_state  = tf.keras.layers.SimpleRNN(32, return_state=True)(x)
    x  = tf.keras.layers.LSTM(32)(x)
    x = tf.keras.layers.Dense(32, activation='relu')(x)
    output_layer = tf.keras.layers.Dense(nb_output, activation='linear')(x)
    model = tf.keras.models.Model(inputs=input_layer, outputs=output_layer)

    return model


class DRQN(tf.keras.Model):
    def __init__(self):
        super(DRQN, self).__init__()
        self.dense1 = tf.keras.layers.Dense(32, activation='relu')
        self.dense2 = tf.keras.layers.Dense(32, activation='relu')
        # self.rnn = tf.keras.layers.SimpleRNN(32, return_state=True)
        self.rnn = tf.keras.layers.LSTM(32, return_state=True)
        self.dense3 = tf.keras.layers.Dense(2, activation='linear')

    def init_hidden(self, batch_size=32):
        # return tf.zeros((batch_size, 4))
        return [tf.zeros((1, batch_size)), tf.zeros((1, batch_size))]

    def call(self, x, hidden):
        x = self.dense1(x)
        # x, hidden_state = self.rnn(x, initial_state=hidden)
        x, h, c  = self.rnn(x, initial_state=hidden)
        x = self.dense2(x)
        x = self.dense3(x)

        # return x, hidden_state
        return x, [h, c]

step_history = []
nb_epsiodes = 500
average_episode_reward = 0

env = gym.make('CartPole-v0')
np.random.seed(123)
env.seed(123)
nb_actions = env.action_space.n
actions = np.arange(nb_actions)
policy = EpsGreedyQPolicy(eps=1., eps_decay_rate=.995, min_eps=.01)
memory = memorys.EpisodeMemory(limit=2000)

# memory = SequentialMemory(limit=50000, maxlen=1)
ini_observation = env.reset().reshape(1, 4)
loss_fn = tf.keras.losses.Huber()
optimizer = tf.keras.optimizers.Adam()
# input_shape = ini_observation.shape
input_shape = (time_step, 4)
model = build_recurrent_q_network(
    input_shape=input_shape,
    nb_output=len(actions))
target_model = build_recurrent_q_network(
    input_shape=input_shape,
    nb_output=len(actions))
# model = DRQN()
# target_model = DRQN()

# inputs = np.random.random([32, 1, 4]).astype(np.float32)
# hidden = tf.zeros((32, 32))
# a, b = model(inputs, hidden)
# a, b = target_model(inputs, hidden)

# inputs = np.random.random([32, 1, 4]).astype(np.float32)
# h = tf.zeros((32, 32))
# c = tf.zeros((32, 32))
# a, b = model(inputs, [h, c])
# a, b = target_model(inputs, [h, c])


# log
import datetime
current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
train_log_dir = 'logs/gradient_tape/' + current_time + '/train'
train_summary_writer = tf.summary.create_file_writer(train_log_dir)

model_dir = 'logs/funcs/' + current_time + '/train'
model_writer = tf.summary.create_file_writer(model_dir)
# tf.summary.trace_on(graph=True, profiler=True)

exp_dir = 'logs/scalar/' + current_time + '/train'
exp_writer = tf.summary.create_file_writer(exp_dir)


agent = DRQNAgent(actions=actions,
                 memory=memory,
                 update_interval=0.01,
                 train_interval=1,
                 batch_size=32,
                 observation=ini_observation,
                 model=model,
                 target_model=target_model,
                 policy=policy,
                 loss_fn=loss_fn,
                 optimizer=optimizer,
                 obs_processor=obs_processor, 
                 is_ddqn=False, 
                 time_step = time_step, 
                 train_log_dir=train_log_dir, 
                 train_summary_writer=train_summary_writer)

from collections import deque
trajectory = deque(maxlen=time_step)

with tqdm.trange(nb_epsiodes) as t:
    for episode in t:
        agent.reset()
        # observation = env.reset().reshape(1, 4)
        observation = env.reset()
        trajectory = np.tile(observation, (time_step, 1))
        agent.observe(trajectory)
        done = False
        step = 0
        episode_reward_history = []
        local_memory = []
        # prev_observation = deepcopy(observation)
        prev_observation = deepcopy(trajectory)
        # train
        while not done:
            action = agent.act()
            observation, reward, done, info = env.step(action)
            step += 1
            trajectory = np.roll(trajectory, -1)
            trajectory[-1] = observation

            episode_reward_history.append(reward)
            agent.observe(trajectory, reward, done)
            # local_memory.append([prev_observation, action, reward,observation, done])
            local_memory.append([prev_observation, action, reward,trajectory, done])
            # prev_observation = deepcopy(observation)
            prev_observation = deepcopy(trajectory)
            if done:
                with exp_writer.as_default():
                    tf.summary.scalar("steps", step, step=episode)
                t.set_description('Episode {}: {} steps'.format(episode, step))
                average_episode_reward = 0.9*average_episode_reward + 0.1*np.sum(episode_reward_history)
                t.set_postfix(episode_reward=average_episode_reward)
                step_history.append(step)
                
                agent.record_episode(local_memory)

                agent.train()
                if episode % 5 == 0:
                    agent.update_target()
                break

        # if last step is bigger than 195, stop the game.
        if all(np.array(step_history[-10:]) >= (env.spec.max_episode_steps - 5)):
            print('Problem is solved in {} episodes.'.format(episode))
            break


# with model_writer.as_default():
#   tf.summary.trace_export(
#       name="my_func_trace",
#       step=0,
#       profiler_outdir=model_dir)

env.close()
x = np.arange(len(step_history))
plt.ylabel('step')
plt.xlabel('episode')
plt.plot(x, step_history)
plt.savefig('result.png')
