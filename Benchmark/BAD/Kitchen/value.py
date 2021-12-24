import tensorflow as tf
import numpy as np

class Value:
    def __init__(self, config, sess):
        self.sess = sess
        self.config = config

        with tf.variable_scope('Value'):
            # BAD state
            self.menu = tf.placeholder(tf.float32, [None, config.num_candidate, config.num_ingredient])
            self.workplace = tf.placeholder(tf.float32, [None, config.num_ingredient])
            # beta
            self.belief = tf.placeholder(tf.float32, [None, config.num_candidate])
            # private features of an individual agent
            self.target = tf.placeholder(tf.float32, [None, config.num_candidate])

            if config.value_architecture == 'mlp':
                self.menu_flat = tf.reshape(self.menu, [-1, config.num_candidate * config.num_ingredient])
                self.in_x = tf.concat([
                    self.belief,
                    self.menu_flat, 
                    self.workplace, 
                    self.target
                ], axis = -1)
                self.fc_layers = [self.in_x]
                for i, dim in enumerate(config.value_fc_dim):
                    if i == len(config.value_fc_dim) - 1:
                        act = None
                    else:
                        act = tf.nn.leaky_relu
                    self.fc_layers.append(tf.layers.dense(self.fc_layers[-1], dim, activation = act))
                self.value = self.fc_layers[-1]
            else:
                raise Exception('Wrong value_architecture type')
            
            assert(self.value.get_shape()[-1] == 1)

            self.value = tf.reshape(self.value, [-1])
            self.value_spvs = tf.placeholder(tf.float32, [None])
            self.td_error = self.value_spvs - self.value
            self.loss = tf.reduce_mean(tf.square(self.td_error))
            self.train_op = tf.train.AdamOptimizer(config.lr).minimize(self.loss)


    def get_value(self, menu, workplace, belief, target):
        value = self.sess.run(
            self.value, 
            feed_dict = {
                self.menu: menu,
                self.workplace: workplace, 
                self.belief: belief, 
                self.target: target
            }
        )

        return value


    def train_value(self, menu, workplace, belief, target, value_spvs):
        loss, td_error, _ = self.sess.run(
            [self.loss, self.td_error, self.train_op], 
            feed_dict = {
                self.menu: menu, 
                self.workplace: workplace, 
                self.belief: belief,
                self.target: target,
                self.value_spvs: value_spvs
            }
        )

        return loss, td_error