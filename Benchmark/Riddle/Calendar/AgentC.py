import tensorflow as tf
import numpy as np
from calendar import Calendar

class Agent:
    def __init__(self, config, name):
        self.config = config
        self.name_ = name
        self.num_obv = config.num_obv
        self.num_action = config.num_action
        self.training = config.training
        self.action_chosen = tf.placeholder(tf.int32, [None])
        self.actions = tf.one_hot(self.action_chosen, self.num_action)
        self.calendar_ = Calendar(self.name_, 'X', -1, -1, -1, self.config.num_slot)  # only need num slots
        # self.global_step = tf.Variable(0, name='global_step', trainable=False)
        # self.epsilon = tf.train.polynomial_decay(
        #     config.epsilon_start,
        #     self.global_step,
        #     config.epsilon_decay,
        #     config.epsilon_min,
        #     power=1.0,
        #     name='epsilon'
        # )

        # self.action_chosen = self.choose_action()
    #     self.epsilon_greedy = EpsilonGreedy(self.num_action)
    #
    # def choose_action(self, q_values, time):
    #     action_chosen, epsilon = self.epsilon_greedy.get_action(q_values, time, self.training)
    #     actions = self.sess.run(tf.one_hot(action_chosen, self.num_action))
    #
    #     return actions, action_chosen, epsilon

    def choose_action(self, sess, q_values, epsilon):

        # With probability epsilon.
        # epsilon = self.sess.run(self.epsilon, feed_dict={
        #     self.global_step: episode
        # })

        #if not self.training:
        #    maximums = np.argwhere(q_values == np.amax(q_values)).flatten().tolist()
        #    action_chosen = int(np.random.choice(maximums, 1))
        if np.random.random() < epsilon:
            # Select a random action.
            action_chosen = np.random.randint(low=0, high=self.num_action)
        else:
            # Otherwise select the action that has the highest Q-value.
            if self.config.soft_max:
                expq = np.exp(self.config.acting_boltzman_beta * (np.squeeze(q_values) - np.max(q_values)))
                act_prob = expq / np.sum(expq)
                if self.config.print_prob:
                    print(act_prob)
                action_chosen = int(np.random.choice(self.num_action, 1, p=act_prob))
            else:
                action_chosen = np.argmax(q_values)

        actions = sess.run(self.actions, feed_dict={
            self.action_chosen: [action_chosen]
        })

        return actions, action_chosen
