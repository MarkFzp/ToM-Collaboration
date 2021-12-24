import tensorflow as tf
import numpy as np

class Value:
    def __init__(self, config, sess):
        self.sess = sess
        self.config = config

        with tf.variable_scope('Value'):
            self.belief_1 = tf.placeholder(tf.float32, [None, config.num_candidate])
            self.belief_2 = tf.placeholder(tf.float32, [None, config.num_candidate])
            self.target_1_idx = tf.placeholder(tf.int32, [None])
            self.target_2_idx = tf.placeholder(tf.int32, [None])
            self.target_1_oh = tf.one_hot(self.target_1_idx, config.num_candidate)
            self.target_2_oh = tf.one_hot(self.target_2_idx, config.num_candidate)

            if config.value_architecture == 'mlp':
                self.in_x = tf.concat([
                    self.belief_1, 
                    self.belief_2, 
                    self.target_1_oh,
                    self.target_2_oh
                ], axis = -1)
                self.fc_layers = [self.in_x]
                for i, dim in enumerate(config.value_fc_dim):
                    if i == len(config.value_fc_dim) - 1:
                        act = None
                    else:
                        act = tf.nn.leaky_relu
                    self.fc_layers.append(tf.layers.dense(self.fc_layers[-1], dim, activation = act))
                self.value = self.fc_layers[-1]
            
            elif config.value_architecture == 'mlp2':
                self.target_1_fcs = [self.target_1_oh]
                self.target_2_fcs = [self.target_2_oh]
                self.cf_1_fcs = [self.belief_1]
                self.cf_2_fcs = [self.belief_2]
                for i, dim in enumerate(config.cf_fc_dim):
                    cf_dense = tf.layers.Dense(dim, activation = tf.nn.leaky_relu)
                    self.target_1_fcs.append(cf_dense(self.target_1_fcs[-1]))
                    self.target_2_fcs.append(cf_dense(self.target_2_fcs[-1]))
                    self.cf_1_fcs.append(cf_dense(self.cf_1_fcs[-1]))
                    self.cf_2_fcs.append(cf_dense(self.cf_2_fcs[-1]))
                
                self.target_1_dense = self.target_1_fcs[-1]
                self.target_2_dense = self.target_2_fcs[-1]
                self.cf_1_dense = self.cf_1_fcs[-1]
                self.cf_2_dense = self.cf_2_fcs[-1]

                self.in_x = tf.concat([
                    self.target_1_dense,
                    self.target_2_dense, 
                    self.cf_1_dense, 
                    self.cf_2_dense
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
            self.train_op = tf.train.AdamOptimizer(config.lr * config.value_lr_increase_factor).minimize(self.loss)


    def get_value(self, belief_1, belief_2, target_1_idx, target_2_idx):
        value = self.sess.run(
            self.value, 
            feed_dict = {
                self.belief_1: belief_1,
                self.belief_2: belief_2, 
                self.target_1_idx: target_1_idx, 
                self.target_2_idx: target_2_idx
            }
        )

        return value


    def train_value(self, belief_1, belief_2, target_1_idx, target_2_idx, value_spvs):
        loss, td_error, _ = self.sess.run(
            [self.loss, self.td_error, self.train_op], 
            feed_dict = {
                self.belief_1: belief_1,
                self.belief_2: belief_2,  
                self.target_1_idx: target_1_idx, 
                self.target_2_idx: target_2_idx,
                self.value_spvs: value_spvs
            }
        )

        return loss, td_error