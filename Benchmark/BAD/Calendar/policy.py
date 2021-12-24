import tensorflow as tf
import numpy as np

class Policy:
    def __init__(self, config, sess, policy_name_appendix, all_calendar):
        self.sess = sess
        self.config = config

        if policy_name_appendix is None:
            self.variable_scope_name = 'Policy'
        else:
            self.variable_scope_name = 'Policy_{}'.format(policy_name_appendix)

        with tf.variable_scope(self.variable_scope_name):
            # public features
            self.all_calendar = tf.constant(all_calendar, dtype = tf.float32)
            # beta
            self.cf_1 = tf.placeholder(tf.float32, [None, config.num_candidate])
            self.cf_2 = tf.placeholder(tf.float32, [None, config.num_candidate])

            # self.batch_size = tf.shape(self.cf_1)[0]

            if config.policy_architecture == 'mlp':
                # self.menu_flat = tf.reshape(self.menu, [-1, config.num_candidate * config.num_ingredient])
                # self.in_x = tf.concat([
                #     self.cf,
                #     self.menu_flat, 
                #     self.workplace
                # ], axis = -1)
                # self.fc_layers = [self.in_x]
                # for i, dim in enumerate(config.policy_fc_dim):
                #     if i == len(config.policy_fc_dim) - 1:
                #         act = None
                #     else:
                #         act = tf.nn.leaky_relu
                #     self.fc_layers.append(tf.layers.dense(self.fc_layers[-1], dim, activation = act))
                # self.logit = self.fc_layers[-1]
                pass

            elif config.policy_architecture == 'context':
                self.pre_context = [self.all_calendar]
                for i, dim in enumerate(config.policy_context_dim):
                    if i == len(config.policy_context_dim) - 1:
                        # act = None
                        self.context_len = dim
                    # else:
                    #     act = tf.nn.leaky_relu
                    self.pre_context.append(tf.layers.dense(self.pre_context[-1], dim, activation = tf.nn.leaky_relu))
                else:
                    self.context_len = self.all_calendar.get_shape()[-1]
                
                self.context = tf.expand_dims(self.pre_context[-1], 0)
                self.belief_mask_1 = tf.stack([self.cf_1] * self.context_len, axis = 2)
                self.belief_mask_2 = tf.stack([self.cf_2] * self.context_len, axis = 2)
                self.context_with_belief_1 = tf.reduce_sum(self.context * self.belief_mask_1, axis = 1)
                self.context_with_belief_2 = tf.reduce_sum(self.context * self.belief_mask_2, axis = 1)

                if config.context_matmul:
                    self.context_with_belief_1 = tf.expand_dims(self.context_with_belief_1, axis = 2)
                    self.context_with_belief_2 = tf.expand_dims(self.context_with_belief_2, axis = 1)
                    self.interconnect = tf.reshape(
                        tf.matmul(self.context_with_belief_1, self.context_with_belief_2), 
                        [-1, self.context_len * self.context_len]
                    )
                else:
                    self.cf_1_fcs = [self.cf_1]
                    self.cf_2_fcs = [self.cf_2]
                    for i, dim in enumerate(config.cf_fc_dim):
                        cf_dense = tf.layers.Dense(dim, activation = tf.nn.leaky_relu)
                        self.cf_1_fcs.append(cf_dense(self.cf_1_fcs[-1]))
                        self.cf_2_fcs.append(cf_dense(self.cf_2_fcs[-1]))
                    self.cf_1_dense = self.cf_1_fcs[-1]
                    self.cf_2_dense = self.cf_2_fcs[-1]
                    self.interconnect = tf.concat([self.context_with_belief_1, self.context_with_belief_2, self.cf_1_dense, self.cf_2_dense], axis = 1)

                self.fc_layers = [self.interconnect]
                for i, dim in enumerate(config.policy_fc_dim):
                    if i == len(config.policy_fc_dim) - 1:
                        act = None
                    else:
                        act = tf.nn.leaky_relu
                    self.fc_layers.append(tf.layers.dense(self.fc_layers[-1], dim, activation = act))
                self.logit = self.fc_layers[-1]
            
            else:
                raise Exception('Wrong policy_architecture type')
            
            assert(self.logit.get_shape()[-1] == config.num_action)
            self.prob_pre = tf.nn.softmax(self.logit)

            if config.explore_method == 'anneal':
                self.uniform_prob = tf.constant(np.ones((config.num_action,)) / config.num_action, dtype = tf.float32)
                self.global_step = tf.Variable(0, name = 'global_step', trainable = False)
                self.epsilon = tf.train.polynomial_decay(
                    config.init_epsilon, 
                    self.global_step, 
                    config.linear_decay_step, 
                    config.end_epsilon, 
                    power = 1.0, 
                    name = 'epsilon'
                )
                self.prob = (1 - self.epsilon) * self.prob_pre + self.epsilon * self.uniform_prob
            elif config.explore_method == 'entropy':
                self.prob = self.prob_pre
            else:
                raise Exception()

            self.argmax_action = tf.argmax(self.prob, axis = 1)
            self.sampled_action = tf.distributions.Categorical(probs = self.prob).sample()

            self.prob_idx = tf.placeholder(tf.int32, [None, 2])
            self.prob_log = tf.log(tf.reshape(
                tf.gather_nd(self.prob, self.prob_idx), 
                [-1, config.num_candidate]
            ))

            self.bad_log_prob = tf.reduce_sum(self.prob_log, axis = 1)
            self.adv = tf.placeholder(tf.float32, [None])

            if config.explore_method == 'anneal':
                self.loss = -1 * tf.reduce_mean(self.bad_log_prob * self.adv)
                self.train_op = tf.train.AdamOptimizer(config.lr).minimize(self.loss, global_step = self.global_step)
            elif config.explore_method == 'entropy':
                self.entropy = -1 * tf.reduce_sum(self.prob * tf.log(self.prob + 1e-12), axis = 1)
                self.loss = -1 * (tf.reduce_mean(self.bad_log_prob * self.adv, axis = 0) + config.entropy_param * tf.reduce_mean(self.entropy, axis = 0))
                self.train_op = tf.train.AdamOptimizer(config.lr).minimize(self.loss)
            


    def sample_action(self, cf_1, cf_2):
        if self.config.test or self.config.change_pair_test:
            action = self.sess.run(
                self.argmax_action, 
                feed_dict = {
                    self.cf_1: cf_1,
                    self.cf_2: cf_2
                }
            )
        else:
            action = self.sess.run(
                self.sampled_action, 
                feed_dict = {
                    self.cf_1: cf_1,
                    self.cf_2: cf_2
                }
            )
        # print(prob_pre)
        # input()

        return action


    def train_policy(self, cf_1, cf_2, prob_idx, adv):
        if self.config.explore_method == 'anneal':
            loss, epsilon, _ = self.sess.run(
                [self.loss, self.epsilon, self.train_op], 
                feed_dict = {
                    self.cf_1: cf_1,
                    self.cf_2: cf_2,
                    self.prob_idx: prob_idx,
                    self.adv: adv
                }
            )
        elif self.config.explore_method == 'entropy':
            epsilon = np.nan
            loss, _ = self.sess.run(
                [self.loss, self.train_op], 
                feed_dict = {
                    self.cf_1: cf_1,
                    self.cf_2: cf_2,
                    self.prob_idx: prob_idx,
                    self.adv: adv
                }
            )
            # print(bad_probs)
            # input()

        return loss, epsilon
