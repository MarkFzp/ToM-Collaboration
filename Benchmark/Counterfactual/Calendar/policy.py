import tensorflow as tf
import numpy as np
import pdb

class Policy:
    def __init__(self, config, sess, policy_name_appendix):
        self.sess = sess
        self.config = config

        if policy_name_appendix is None:
            self.variable_scope_name = 'Policy'
        else:
            self.variable_scope_name = 'Policy_{}'.format(policy_name_appendix)


        with tf.variable_scope(self.variable_scope_name):
            # batch_size * seq_len * in_x_dim
            self.in_x = tf.placeholder(tf.float32, [None, None, config.in_x_dim])
            self.in_state = tf.placeholder_with_default(tf.zeros([tf.shape(self.in_x)[0], config.policy_hidden_dim]), [None, config.policy_hidden_dim])
            
            self.in_layers = [self.in_x]
            for in_fc_dim in config.in_fc_dims:
                self.in_layers.append(tf.layers.dense(self.in_layers[-1], in_fc_dim, activation = tf.nn.leaky_relu))
            self.in_x_fc = self.in_layers[-1]

            self.cell = tf.keras.layers.GRU(config.policy_hidden_dim, return_sequences = True, return_state = True, name = 'gru_2')
            self.out_seq, self.out_state = self.cell(self.in_x_fc, initial_state = self.in_state)
            
            self.out_layers = [self.out_seq]
            for idx, out_fc_dim in enumerate(config.out_fc_dims):
                if idx == len(config.out_fc_dims) - 1:
                    activation = None
                else:
                    activation = tf.nn.leaky_relu
                self.out_layers.append(tf.layers.dense(self.out_layers[-1], out_fc_dim, activation = activation))
            self.out_logit = self.out_layers[-1]
            assert(self.out_logit.get_shape()[-1] == config.num_action)

            self.action_prob = tf.nn.softmax(self.out_logit, axis = -1)
            self.sampled_action = tf.distributions.Categorical(probs = self.action_prob).sample()
            self.argmax_action = tf.argmax(self.action_prob, axis = -1)

            self.log_prob_idx = tf.placeholder(tf.int32, [None, 3])
            self.action_prob_selected = tf.gather_nd(self.action_prob, self.log_prob_idx)
            self.adv = tf.placeholder(tf.float32, [None])
            
            self.entropy_prob_idx = self.log_prob_idx[:, :2]
            self.entropy_prob_selected = tf.gather_nd(self.action_prob, self.entropy_prob_idx)
            self.entropy = -1 * tf.reduce_sum(self.entropy_prob_selected * tf.log(self.entropy_prob_selected + 1e-12), axis = 1)
            self.loss = -1 * (tf.reduce_mean(tf.log(self.action_prob_selected) * self.adv) + config.entropy_param * tf.reduce_mean(self.entropy, axis = 0))
            self.train_op = tf.train.AdamOptimizer(config.lr).minimize(self.loss)




    def sample_action(self, in_x, in_state_2d):
        in_x = in_x[np.newaxis, np.newaxis, :]
    
        if self.config.test or self.config.change_pair_test or self.config.max_len_msg_test:
            action, action_prob, out_state_2d = self.sess.run(
                [self.argmax_action, self.action_prob, self.out_state], 
                feed_dict = {
                    self.in_x: in_x,
                    self.in_state: in_state_2d
                } if in_state_2d is not None else {
                    self.in_x: in_x
                }
            )
        else:
            action, action_prob, out_state_2d = self.sess.run(
                [self.sampled_action, self.action_prob, self.out_state], 
                feed_dict = {
                    self.in_x: in_x,
                    self.in_state: in_state_2d
                } if in_state_2d is not None else {
                    self.in_x: in_x
                }
            )

        action = action[0, 0]
        action_prob = action_prob[0, 0]

        return action, action_prob, out_state_2d

    
    

    def train(self, in_x, log_prob_idx, adv):
        loss, _ = self.sess.run(
            [self.loss, self.train_op], 
            feed_dict = {
                self.in_x: in_x,
                self.log_prob_idx: log_prob_idx,
                self.adv: adv
            }
        )

        return loss
