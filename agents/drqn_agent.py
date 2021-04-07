import numpy as np
import tensorflow as tf
import copy


class DRQNAgent():
    """
        Deep Recurrent Q Learning Agent
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
        # self.warmup_steps = warmup_steps
        self.warmup_steps = 3
        self.model = model
        self.target_model = target_model
        self.train_interval = train_interval
        # self.update_interval = update_interval
        self.update_interval = 0.1
        self.is_ddqn = is_ddqn
        self.time_step = time_step

        self.loss_fn = loss_fn
        self.optimizer = optimizer

        self.step = 0

        self.train_loss = tf.keras.metrics.Mean('train_loss', dtype=tf.float32)
        import datetime
        # current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.train_log_dir = train_log_dir
        self.train_summary_writer = train_summary_writer

        # self.hidden_state = tf.zeros((1, 32))
        # self.hidden_state = [tf.zeros((1, 32)), tf.zeros((1, 32))]


    def act(self):
        # self.target_model.layers[2].states[0] = self.hidden_state
        # q_values, hidden_state = self.target_model(np.array([self.observation]))
        # q_values, hidden_state = self.target_model(np.array([self.observation]), self.hidden_state, training=False)
        # self.hidden_state = hidden_state

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

    def record_episode(self, episode):
        self.memory.append(episode)

    def train(self):
        if self.step > self.warmup_steps \
                and self.step % self.train_interval == 0:

            for epoch in range(32):
                self._experience_replay()


        self.step += 1

    def update_target(self):
        self.target_model.set_weights(self.model.get_weights())
        # if self.update_interval > 1:
        #     self._hard_update_target_model()
        # else:
        #     self._soft_update_target_model()

    def _experience_replay(self):
        episodes = self.memory.sample(self.batch_size, time_step=10)
        for episode in episodes:
            state0_batch, action_batch, reward_batch, state1_batch, terminal_batch  = episode

            reward_batch = reward_batch.reshape(-1, 1)
            terminal_batch = terminal_batch.reshape(-1, 1)

            # hidden_state = self.target_model.init_hidden()
            # target_q_values, _ = self.target_model(state1_batch, tf.zeros((len(state1_batch), 32)))
            # target_q_values, _ = self.target_model(state1_batch, hidden_state )
            target_q_values = self.target_model(state1_batch)
            # target_q_values, _ = self.target_model(state1_batch)
            # target_q_values = []
            # for state1 in state1_batch:
            #     state1 = np.expand_dims(state1, axis=0)
            #     target, hidden_state = self.target_model(state1, hidden_state, training=False)
            #     target_q_values.append(tf.squeeze(target))
            # target_q_values = tf.convert_to_tensor(target_q_values)

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

            with self.train_summary_writer.as_default():
                tf.summary.scalar("loss", self.train_loss.result(), step=self.step)
                tf.summary.scalar("epsilon", self.policy.eps, step=self.step)

            self.model.reset_states()
            self.target_model.reset_states()


    # @tf.function
    def _train_on_batch(self, states, masks, targets):
        with tf.GradientTape() as tape:
            # hidden_state = self.model.init_hidden(states.shape[0])
            # y_preds, _ = self.model(states, tf.zeros((len(states), 32)))
            y_preds = self.model(states)

            # y_preds = []
            # hidden_state = tf.zeros((1, 32))
            # hidden_state = self.model.init_hidden()
            # for state in states:
            #     state = np.expand_dims(state, axis=0)
            #     target, hidden_state = self.model(state, hidden_state, training=True)
            #     y_preds.append(tf.squeeze(target))
            # y_preds = tf.convert_to_tensor(y_preds)

            y_preds = tf.math.multiply(y_preds, masks)
            loss_value = self.loss_fn(targets, y_preds)

        grads = tape.gradient(loss_value, self.model.trainable_weights)

        self.optimizer.apply_gradients(
            zip(grads, self.model.trainable_weights))

        self.train_loss(loss_value)

    def _hard_update_target_model(self):
        """ for hard update """
        # if self.step % self.update_interval == 0:
        #     self.target_model.set_weights(self.model.get_weights())
        self.target_model.set_weights(self.model.get_weights())

    def _soft_update_target_model(self):
        target_model_weights = np.array(self.target_model.get_weights())
        model_weights = np.array(self.model.get_weights())
        new_weight = (1. - self.update_interval) * target_model_weights \
            + self.update_interval * model_weights
        self.target_model.set_weights(new_weight)

    def reset(self):
        # self.observation = None
        # self.prev_observation = None
        self.model.reset_states()
        self.target_model.reset_states()
        # self.hidden_state = tf.zeros((1, 32))
        # self.hidden_state = [tf.zeros((1, 32)), tf.zeros((1, 32))]
        # self.hidden_state = self.target_model.init_hidden(1)
        # self.hidden_state = self.target_model.init_hidden(1)
