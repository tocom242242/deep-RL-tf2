import numpy as np
import tensorflow as tf
import copy
import datetime


class DRQNAgent():
    """
        Deep Recurrent Q Learning Agent
    """

    def __init__(self,
                 training=True,
                 policy=None,
                 epochs=32,
                 gamma=.99,
                 actions=None,
                 memory=None,
                 memory_interval=1,
                 model=None,
                 target_model=None,
                 batch_size=32,
                 observation=None,
                 obs_processor=None,
                 loss_fn=None,
                 optimizer=None,
                 is_ddqn=False,
                 time_step=4,
                 train_loss=None,
                 train_log_dir=None,
                 train_summary_writer=None,
                 ):

        self.training = training
        self.policy = policy
        self.actions = actions
        self.gamma = gamma
        self.obs_processor = obs_processor
        self.observation = self.obs_processor(observation)
        self.prev_observation = None
        self.memory = memory
        self.memory_interval = memory_interval
        self.batch_size = batch_size
        self.epochs = epochs
        self.model = model
        self.target_model = target_model
        self.is_ddqn = is_ddqn
        self.time_step = time_step

        self.loss_fn = loss_fn
        self.optimizer = optimizer

        self.step = 0

        self.train_loss = tf.keras.metrics.Mean('train_loss', dtype=tf.float32)
        self.train_log_dir = train_log_dir
        self.train_summary_writer = train_summary_writer

    def act(self):
        q_values = self.target_model(np.array([self.observation]))

        q_values = tf.squeeze(q_values)
        action_id = self.policy.select_action(
            q_values=q_values, is_training=self.training)
        self.recent_action_id = action_id
        action = self.actions[action_id]

        self.policy.decay_eps_rate()
        return action

    def observe(self, observation, reward=None, is_terminal=None):
        self.prev_observation = copy.deepcopy(self.observation)
        self.observation = observation
        self.step += 1

    def record_episode(self, episode):
        self.memory.append(episode)

    def train(self):
        for epoch in range(self.epochs):
            self._experience_replay()

    def update_target_hard(self):
        self.target_model.set_weights(self.model.get_weights())

    def update_target_soft(self, tau=0.001):
        target_model_weights = np.array(self.target_model.get_weights())
        model_weights = np.array(self.model.get_weights())
        new_weight = (1. - tau) * target_model_weights + tau * model_weights
        self.target_model.set_weights(new_weight)

    def _experience_replay(self):
        episodes = self.memory.sample(
            self.batch_size, time_step=self.time_step)
        for episode in episodes:
            state0_batch, action_batch, reward_batch, state1_batch, terminal_batch = episode

            reward_batch = reward_batch.reshape(-1, 1)
            terminal_batch = terminal_batch.reshape(-1, 1)

            target_q_values = self.target_model(state1_batch)
            discounted_reward_batch = self.gamma * target_q_values * terminal_batch
            targets = reward_batch + discounted_reward_batch

            targets_one_hot = np.zeros((len(targets), len(self.actions)))
            if self.is_ddqn:
                q_values = self.model(state1_batch)
                argmax_actions = np.argmax(q_values, axis=1)
                for idx, (action, argmax_action) in enumerate(
                        zip(action_batch, argmax_actions)):
                    targets_one_hot[idx][action] = targets[idx][argmax_action]
            else:
                for idx, action in enumerate(action_batch):
                    targets_one_hot[idx][action] = max(targets[idx])

            mask = tf.one_hot(action_batch, len(self.actions))
            state0_batch = tf.convert_to_tensor(state0_batch)

            self._train_on_batch(state0_batch, mask, targets_one_hot)

            with self.train_summary_writer.as_default():
                tf.summary.scalar(
                    "loss", self.train_loss.result(), step=self.step)
                tf.summary.scalar("epsilon", self.policy.eps, step=self.step)

            self.model.reset_states()
            self.target_model.reset_states()

    # @tf.function
    def _train_on_batch(self, states, masks, targets):
        with tf.GradientTape() as tape:
            y_preds = self.model(states)
            y_preds = tf.math.multiply(y_preds, masks)
            loss_value = self.loss_fn(targets, y_preds)

        grads = tape.gradient(loss_value, self.model.trainable_weights)

        self.optimizer.apply_gradients(
            zip(grads, self.model.trainable_weights))

        self.train_loss(loss_value)

    def reset(self):
        self.model.reset_states()
        self.target_model.reset_states()
