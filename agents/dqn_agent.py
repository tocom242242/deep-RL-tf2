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
                 epochs=32,
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
                 optimizer=None, 
                 is_ddqn=False):

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
        self.is_ddqn = is_ddqn
        self.epochs = epochs

        self.loss_fn = loss_fn
        self.optimizer = optimizer

        self.step = 0

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

        if self.training and reward is not None:
            self.memory.append(self.prev_observation,
                               self.recent_action_id,
                               reward,
                               observation, 
                               terminal=is_terminal)

        self.step += 1


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
        state0_batch, action_batch, reward_batch, state1_batch, terminal_batch  = self.memory.sample(self.batch_size)

        reward_batch = reward_batch.reshape(-1, 1)
        terminal_batch = terminal_batch.reshape(-1, 1)

        target_q_values = self.target_model(state1_batch)

        discounted_reward_batch = self.gamma * target_q_values * terminal_batch
        targets = reward_batch + discounted_reward_batch

        targets_one_hot = np.zeros((len(targets), len(self.actions)))
        if self.is_ddqn:
            q_values = self.model(state1_batch)
            argmax_actions = np.argmax(q_values, axis=1)
            for idx, (action, argmax_action) in enumerate(zip(action_batch, argmax_actions)):
                targets_one_hot[idx][action] = targets[idx][argmax_action]
        else:
            for idx, action in enumerate(action_batch):
                targets_one_hot[idx][action] = max(targets[idx])

        mask = tf.one_hot(action_batch, len(self.actions))
        state0_batch = tf.convert_to_tensor(state0_batch)

        self._train_on_batch(state0_batch, mask, targets_one_hot)

    # @tf.function
    def _train_on_batch(self, states, masks, targets):
        with tf.GradientTape() as tape:
            y_preds = self.model(states)
            y_preds = tf.math.multiply(y_preds, masks)
            loss_value = self.loss_fn(targets, y_preds)

        grads = tape.gradient(loss_value, self.model.trainable_weights)

        self.optimizer.apply_gradients(
            zip(grads, self.model.trainable_weights))

    def reset(self):
        self.observation = None
        self.prev_observation = None
