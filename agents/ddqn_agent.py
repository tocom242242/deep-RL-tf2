import numpy as np
import tensorflow as tf
import copy


def build_model(input_shape, nb_output):
    input_layer = tf.keras.layers.Input(shape=input_shape)
    x = tf.keras.layers.Flatten()(input_layer)
    x = tf.keras.layers.Dense(32, activation="relu")(x)
    x = tf.keras.layers.Dense(32, activation="relu")(x)
    output_layer = tf.keras.layers.Dense(nb_output, activation="linear")(x)
    model = tf.keras.models.Model(inputs=input_layer, outputs=output_layer)

    return model


class DDQNAgent():
    """
        Double Deep Q Network Agent
    """

    def __init__(self,
                 training=True,
                 policy=None,
                 gamma=.99,
                 learning_rate=.001,
                 actions=None,
                 memory=None,
                 memory_interval=1,
                 update_interval=100,
                 train_interval=1,
                 batch_size=32,
                 nb_steps_warmup=200,
                 observation=None,
                 input_shape=None,
                 obs_processer=None):

        self.training = training
        self.policy = policy
        self.actions = actions
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.obs_processer = obs_processer
        self.recent_action_id = None
        self.recent_observation = self.obs_processer(observation)
        self.previous_observation = None
        self.memory = memory
        self.memory_interval = memory_interval
        self.batch_size = batch_size
        self.nb_steps_warmup = nb_steps_warmup
        self.model = build_model(input_shape, len(self.actions))
        self.target_model = build_model(input_shape, len(self.actions))
        self.nb_actions = len(self.actions)
        self.train_interval = train_interval
        self.step = 0
        self.trainable_model = None
        self.update_interval = update_interval

    def compile(self):
        mask = tf.keras.layers.Input(name="mask", shape=(self.nb_actions, ))
        output = tf.keras.layers.multiply([self.model.output, mask])
        trainable_model = tf.keras.models.Model(
            inputs=[self.model.input, mask],
            outputs=[output])
        optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        trainable_model.compile(loss="mse", optimizer=optimizer)
        self.trainable_model = trainable_model

    def act(self):
        action_id = self.forward()
        action = self.actions[action_id]
        return action

    def forward(self):
        q_values = self.compute_q_values(self.recent_observation)
        action_id = self.policy.select_action(
            q_values=q_values, is_training=self.training)
        self.recent_action_id = action_id

        return action_id

    def observe(self, observation, reward=None, is_terminal=None):
        self.previous_observation = copy.deepcopy(self.recent_observation)
        self.recent_observation = self.obs_processer(observation)

        if self.training and reward is not None:
            if self.step % self.memory_interval == 0:
                self.memory.append(self.previous_observation,
                                   self.recent_action_id,
                                   reward,
                                   terminal=is_terminal)
            self.experience_replay()
            self.policy.decay_eps_rate()

        self.step += 1

    def experience_replay(self):
        if self.step > self.nb_steps_warmup \
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

            target_batch = np.zeros((self.batch_size, len(self.actions)))
            reward_batch = np.array(reward_batch)
            q_values = self.predict_on_batch_by_model(state1_batch)
            argmax_actions = np.argmax(q_values, axis=1)
            target_q_values = self.predict_on_batch_by_target(state1_batch)
            double_q_values = []
            for a, t in zip(argmax_actions, target_q_values):
                double_q_values.append(t[a])
            double_q_values = np.array(double_q_values)
            discounted_reward_batch = (self.gamma * double_q_values)
            discounted_reward_batch *= terminal_batch

            targets = reward_batch + discounted_reward_batch
            mask = np.zeros((len(action_batch), len(self.actions)))
            target_batch = np.zeros((self.batch_size, len(self.actions)))
            for idx, (action, target) in enumerate(zip(action_batch, targets)):
                target_batch[idx][action] = target
                mask[idx][action] = 1.

            self.train_on_batch(state0_batch,
                                mask,
                                target_batch)

        if self.update_interval > 1:
            # hard update
            self.update_target_model_hard()
        else:
            # soft update
            self.update_target_model_soft()

    def train_on_batch(self, state_batch, mask, targets):
        state_batch = np.array(state_batch)
        self.trainable_model.train_on_batch([state_batch, mask],
                                            [targets])

    def predict_on_batch_by_model(self, state1_batch):
        state1_batch = np.array(state1_batch)
        q_values = self.model.predict(state1_batch)
        return q_values

    def predict_on_batch_by_target(self, state1_batch):
        state1_batch = np.array(state1_batch)
        q_values = self.target_model.predict(state1_batch)
        return q_values

    def compute_q_values(self, state):
        q_values = self.target_model.predict(np.array([state]))
        return q_values[0]

    def update_target_model_hard(self):
        """ for hard update """
        if self.step % self.update_interval == 0:
            self.target_model.set_weights(self.model.get_weights())

    def update_target_model_soft(self):
        target_model_weights = np.array(self.target_model.get_weights())
        model_weights = np.array(self.model.get_weights())
        new_weight = (1. - self.update_interval) * target_model_weights \
            + self.update_interval * model_weights
        self.target_model.set_weights(new_weight)

    def reset(self):
        self.recent_observation = None
        self.previous_observation = None
        self.recent_action_id = None
