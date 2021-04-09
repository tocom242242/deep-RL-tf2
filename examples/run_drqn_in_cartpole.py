import datetime
import tqdm
import gym
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from copy import deepcopy

# import sys
# sys.path.append('..')
# import agents.memorys as memorys
# from agents.policies import EpsGreedyQPolicy
# from agents.drqn_agent import DRQNAgent

from drqn_agent import DRQNAgent
from policies import EpsGreedyQPolicy
import memorys as memorys


def obs_processor(raw_obs):
    """convert raw observation to agent's observation"""
    return raw_obs


time_step = 10
nb_epsiodes = 200

np.random.seed(123)

env = gym.make('CartPole-v0')
env.seed(123)
nb_actions = env.action_space.n
actions = np.arange(nb_actions)
policy = EpsGreedyQPolicy(eps=1., eps_decay_rate=.995, min_eps=.01)
memory = memorys.EpisodeMemory(limit=2000)

ini_observation = env.reset()
loss_fn = tf.keras.losses.Huber()
optimizer = tf.keras.optimizers.Adam()

input_shape = (time_step, ini_observation.shape[0])

# log
current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
train_log_dir = 'logs/gradient_tape/' + current_time + '/train'
train_summary_writer = tf.summary.create_file_writer(train_log_dir)

exp_dir = 'logs/scalar/' + current_time + '/train'
exp_writer = tf.summary.create_file_writer(exp_dir)

step_history = []
average_episode_reward = 0


def build_recurrent_q_network(input_shape, nb_output):
    input_layer = tf.keras.layers.Input(shape=input_shape)
    x = tf.keras.layers.Dense(32, activation='relu')(input_layer)
    x = tf.keras.layers.LSTM(32)(x)
    x = tf.keras.layers.Dense(32, activation='relu')(x)
    output_layer = tf.keras.layers.Dense(nb_output, activation='linear')(x)
    model = tf.keras.models.Model(inputs=input_layer, outputs=output_layer)

    return model


model = build_recurrent_q_network(
    input_shape=input_shape,
    nb_output=len(actions))
target_model = build_recurrent_q_network(
    input_shape=input_shape,
    nb_output=len(actions))


agent = DRQNAgent(actions=actions,
                  memory=memory,
                  epochs=32,
                  batch_size=32,
                  observation=ini_observation,
                  model=model,
                  target_model=target_model,
                  policy=policy,
                  loss_fn=loss_fn,
                  optimizer=optimizer,
                  obs_processor=obs_processor,
                  is_ddqn=False,
                  time_step=time_step,
                  train_log_dir=train_log_dir,
                  train_summary_writer=train_summary_writer)


with tqdm.trange(nb_epsiodes) as t:
    for episode in t:
        agent.reset()
        observation = env.reset()
        trajectory = np.tile(observation, (time_step, 1))
        agent.observe(trajectory)
        prev_observation = deepcopy(trajectory)
        done = False
        step = 0
        episode_reward_history = []
        local_memory = []

        while not done:
            action = agent.act()
            observation, reward, done, info = env.step(action)
            step += 1

            trajectory = np.roll(trajectory, -1)
            trajectory[-1] = observation

            episode_reward_history.append(reward)
            agent.observe(trajectory, reward, done)
            local_memory.append(
                [prev_observation, action, reward, trajectory, done])
            prev_observation = deepcopy(trajectory)
            if done:
                with exp_writer.as_default():
                    tf.summary.scalar("steps", step, step=episode)
                t.set_description('Episode {}: {} steps'.format(episode, step))
                average_episode_reward = 0.9 * average_episode_reward + \
                    0.1 * np.sum(episode_reward_history)
                t.set_postfix(episode_reward=average_episode_reward)
                step_history.append(step)

                agent.record_episode(local_memory)
                if episode > 3:
                    agent.train()
                    if episode % 5 == 0:
                        agent.update_target_hard()
                break

        # if last step is bigger than 195, stop the game.
        if all(np.array(step_history[-10:]) >=
               (env.spec.max_episode_steps - 5)):
            print('Problem is solved in {} episodes.'.format(episode))
            break


env.close()
x = np.arange(len(step_history))
plt.ylabel('step')
plt.xlabel('episode')
plt.plot(x, step_history)
plt.savefig('result.png')
