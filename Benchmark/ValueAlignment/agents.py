import tensorflow as tf
import numpy as np
from Benchmark.ValueAlignment.belief import *


class Robot:
    def __init__(self, config):
        self.config = config
        self.belief = RobotBelief(config.num_goals, config.num_hactions, config.num_ractions)

        self.qvalues_ph = tf.placeholder(tf.float32, [None, config.num_hactions, config.num_ractions, config.num_goals], 'qvalues')
        self.haction_prob_ph = tf.placeholder(tf.float32, [None, config.num_hactions, config.num_ractions, config.num_goals], 'human_action_probs')
        self.belief_ph = tf.placeholder(tf.float32, [None, config.num_goals], 'belief')

        self.act_opt = self._act_opt()

    def _act_opt(self):
        s = self.qvalues_ph * self.haction_prob_ph * self.belief_ph[:,tf.newaxis,tf.newaxis,:]
        q = tf.reduce_sum(s, axis=(1, -1))

        m = tf.argmax(q, axis=-1)

        return tf.one_hot(m, self.config.num_ractions)

    def init_belief(self, sess):
        self.belief.init_belief(sess)

    def act(self, sess, qvalues, hprob, belief, train=True, epsilon=None):

        if train:
            s = np.random.choice([0,1], p=[epsilon, 1-epsilon])
            if s == 0:
                ## random
                bs = hprob.shape[0]
                a = np.random.choice(self.config.num_ractions, size=(bs, ))

                return np.eye(self.config.num_ractions)[a]
            else:
                return sess.run(self.act_opt, {self.qvalues_ph: qvalues,
                                               self.haction_prob_ph: hprob,
                                               self.belief_ph: belief})
        else:
            return sess.run(self.act_opt, {self.qvalues_ph: qvalues,
                                       self.haction_prob_ph: hprob,
                                           self.belief_ph: belief})


class Human:
    def __init__(self, config):
        self.config = config

        self.qvalues_ph = tf.placeholder(tf.float32,
                                         [None, config.num_hactions, config.num_goals],
                                         'qvalues')

        self.target_goal_ph = tf.placeholder(tf.int64, [None,], 'goal')

        self.temp_ph = tf.placeholder(tf.float32, [], 'beta')

        self.act_opt = self._act_opt()


    def _act_opt(self):
        bind = tf.cast(tf.range(tf.shape( self.target_goal_ph)[0]), tf.int64)
        indexed_logits = tf.gather_nd(tf.transpose(self.qvalues_ph / self.temp_ph, [0,2,1]), tf.stack([bind, self.target_goal_ph], axis=1))

        act = tf.random.multinomial(indexed_logits, 1)[:,0]

        act = tf.one_hot(act, self.config.num_hactions)
        return act


    def act(self, sess, qvalues, goal, temp, train=True):
        # self.beta = min(self.beta * 1.000001, self.config.beta)
        return sess.run(self.act_opt, {self.qvalues_ph: qvalues,
                                       self.target_goal_ph: goal,
                                       self.temp_ph: temp})


class HumanEpislonGreedy:
    def __init__(self, config):
        self.config = config
        self.beta = self.config.beta

        self.qvalues_ph = tf.placeholder(tf.float32,
                                         [None, config.num_hactions, config.num_goals],
                                         'qvalues')

        self.target_goal_ph = tf.placeholder(tf.int64, [None,], 'goal')

        self.act_opt = self._act_opt()


    def _act_opt(self):
        bind = tf.cast(tf.range(tf.shape( self.target_goal_ph)[0]), tf.int64)
        indexed_logits = tf.gather_nd(tf.transpose(self.qvalues_ph * self.beta, [0,2,1]), tf.stack([bind, self.target_goal_ph], axis=1))
        #
        # act = tf.random.multinomial(indexed_logits, 1)[:,0]

        act = tf.argmax(indexed_logits, axis=-1)

        act = tf.one_hot(act, self.config.num_hactions)
        return act

    def act(self, sess, qvalues, target, train=True, epsilon=None):

        if train:
            s = np.random.choice([0,1], p=[epsilon, 1-epsilon])
            if s == 0:
                ## random
                bs = qvalues.shape[0]
                a = np.random.choice(self.config.num_hactions, size=(bs,))

                return np.eye(self.config.num_hactions)[a]

            else:

                return sess.run(self.act_opt, {self.qvalues_ph: qvalues,
                                               self.target_goal_ph: target})
        else:
            return sess.run(self.act_opt, {self.qvalues_ph: qvalues,
                                           self.target_goal_ph: target})