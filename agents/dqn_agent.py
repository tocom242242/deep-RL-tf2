import numpy as np
import tensorflow as tf
import copy


class DQNAgent():
    """
        Deep Q Network Agent
    """

    def __init__(self,
                 training=True,
                 policy=None,
                 gamma=.99,
                 actions=None,
                 memory=None,
                 memory_interval=1,
                 model=None,
                 target_model=None,
                 update_interval=100,
                 train_interval=1,
                 batch_size=32,
                 warmup_steps=200,
                 observation=None,
                 obs_processor=None,
                 loss_fn=None,
                 optimizer=None):

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
        self.warmup_steps = warmup_steps
        self.model = model
        self.target_model = target_model
        self.train_interval = train_interval
        self.update_interval = update_interval

        self.loss_fn = loss_fn
        self.optimizer = optimizer

        self.step = 0

    def act(self):
        action_id = self._forward()
        action = self.actions[action_id]
        return action

    def observe(self, observation, reward=None, is_terminal=None):
        self.prev_observation = copy.deepcopy(self.observation)
        self.observation = observation

        if self.training and reward is not None:
            if self.step % self.memory_interval == 0:
                self.memory.append(self.prev_observation,
                                   self.recent_action_id,
                                   reward,
                                   terminal=is_terminal)
            self._experience_replay()
            self.policy.decay_eps_rate()

        self.step += 1

    def _forward(self):
        q_values = self._compute_q_values(self.observation)
        action_id = self.policy.select_action(
            q_values=q_values, is_training=self.training)
        self.recent_action_id = action_id

        return action_id

    def _experience_replay(self):
        if self.step > self.warmup_steps \
                and self.step % self.train_interval == 0:

            experiences = self.memory.sample(self.batch_size)

            state0_batch = []
            reward_batch = []
            action_batch = []
            state1_batch = []
            terminal_batch = []

            for e in experiences:
                state0_batch.append(e.state0)
                state1_batch.append(e.state1)
                reward_batch.append(e.reward)
                action_batch.append(e.action)
                terminal_batch.append(0. if e.terminal else 1.)

            reward_batch = np.array(reward_batch).reshape(-1, 1)
            terminal_batch = np.array(terminal_batch).reshape(-1, 1)

            target_q_values = self._predict_on_batch(state1_batch)

            discounted_reward_batch = self.gamma * target_q_values * terminal_batch
            targets = reward_batch + discounted_reward_batch

            mask = tf.one_hot(action_batch, 2)
            targets = tf.math.multiply(targets, mask)
            state0_batch = tf.convert_to_tensor(state0_batch)

            self._train_on_batch(state0_batch, mask, targets)

        if self.update_interval > 1:
            # hard update
            self._hard_update_target_model()
        else:
            # soft update
            self._soft_update_target_model()

    @tf.function
    def _train_on_batch(self, states, masks, targets):
        with tf.GradientTape() as tape:
            y_preds = self.model(states)
            y_preds = tf.math.multiply(y_preds, masks)
            loss_value = self.loss_fn(targets, y_preds)

        grads = tape.gradient(loss_value, self.model.trainable_weights)

        self.optimizer.apply_gradients(
            zip(grads, self.model.trainable_weights))

    def _predict_on_batch(self, state1_batch):
        state1_batch = np.array(state1_batch)
        q_values = self.target_model.predict(state1_batch)
        return q_values

    def _compute_q_values(self, state):
        q_values = self.target_model.predict(np.array([state]))
        return q_values[0]

    def _hard_update_target_model(self):
        """ for hard update """
        if self.step % self.update_interval == 0:
            self.target_model.set_weights(self.model.get_weights())

    def _soft_update_target_model(self):
        target_model_weights = np.array(self.target_model.get_weights())
        model_weights = np.array(self.model.get_weights())
        new_weight = (1. - self.update_interval) * target_model_weights \
            + self.update_interval * model_weights
        self.target_model.set_weights(new_weight)

    def reset(self):
        self.observation = None
        self.prev_observation = None
