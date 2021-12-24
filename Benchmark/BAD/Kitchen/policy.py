import tensorflow as tf
import numpy as np
 
class Policy:
    def __init__(self, config, sess, policy_name_appendix):
        self.sess = sess
        self.config = config

        if policy_name_appendix is None:
            self.variable_scope_name = 'Policy'
        else:
            self.variable_scope_name = 'Policy_{}'.format(policy_name_appendix)

        with tf.variable_scope(self.variable_scope_name):
            # public features
            self.menu = tf.placeholder(tf.float32, [None, config.num_candidate, config.num_ingredient])
            self.workplace = tf.placeholder(tf.float32, [None, config.num_ingredient])
            # beta
            # self.belief = tf.placeholder(tf.float32, [None, config.num_candidate])
            # private features of an individual agent
            # self.target = tf.placeholder(tf.float32, [None, config.num_candidate])
            
            # counterfactual belief or target
            self.cf = tf.placeholder(tf.float32, [None, config.num_candidate])

            self.batch_size = tf.shape(self.menu)[0]

            if config.policy_architecture == 'mlp':
                self.menu_flat = tf.reshape(self.menu, [-1, config.num_candidate * config.num_ingredient])
                self.in_x = tf.concat([
                    self.cf,
                    self.menu_flat, 
                    self.workplace
                ], axis = -1)
                self.fc_layers = [self.in_x]
                for i, dim in enumerate(config.policy_fc_dim):
                    if i == len(config.policy_fc_dim) - 1:
                        act = None
                    else:
                        act = tf.nn.leaky_relu
                    self.fc_layers.append(tf.layers.dense(self.fc_layers[-1], dim, activation = act))
                self.logit = self.fc_layers[-1]

            elif config.policy_architecture == 'context':
                self.pre_context = [self.menu]
                for i, dim in enumerate(config.policy_context_dim):
                    context_fc_dim, with_context_fc_dim = dim
                    if i == len(config.policy_context_dim) - 1:
                        self.context_len = with_context_fc_dim
                    pre_context = self.pre_context[-1]
                    
                    context = tf.reduce_sum(pre_context, axis = 1)
                    context_fc = tf.layers.dense(context, context_fc_dim, activation = tf.nn.leaky_relu)
                    context_fc_stack = tf.stack([context_fc] * config.num_candidate, axis = 1)
                    
                    with_context = tf.concat([pre_context, context_fc_stack], axis = 2)
                    with_context_fc = tf.layers.dense(with_context, with_context_fc_dim, activation = tf.nn.leaky_relu)
                    self.pre_context.append(with_context_fc)
                
                self.context = self.pre_context[-1]
                self.belief_mask = tf.stack([self.cf] * self.context_len, axis = 2)
                self.context_with_belief = tf.reduce_sum(self.context * self.belief_mask, axis = 1)
                self.context_with_belief = tf.expand_dims(self.context_with_belief, axis = 2)

                self.workplace_fc = tf.layers.dense(self.workplace, config.workplace_fc_dim, activation = tf.nn.leaky_relu)
                self.workplace_fc = tf.expand_dims(self.workplace_fc, axis = 1)

                self.interconnect = tf.reshape(
                    tf.matmul(self.context_with_belief, self.workplace_fc), 
                    [self.batch_size, self.context_len * config.workplace_fc_dim]
                )

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

            # if config.test or config.change_pair_test or config.order_free_test:
            #     self.sampled_action = tf.argmax(self.prob, axis = 1)
            # else:
            #     self.sampled_action = tf.distributions.Categorical(probs = self.prob).sample()
            self.argmax_action = tf.argmax(self.prob, axis = 1)
            self.sampled_action = tf.distributions.Categorical(probs = self.prob).sample()

            self.tea_prob_idx = tf.placeholder(tf.int32, [None, 2])
            self.stu_prob_idx = tf.placeholder_with_default(tf.constant([[-1, -1]], dtype = tf.int32), [None, 2])
            self.stu_prob_idx_fed = tf.placeholder(tf.bool, shape = [])
            
            self.tea_prob_all = tf.reshape(
                tf.gather_nd(self.prob, self.tea_prob_idx), 
                [-1, config.num_candidate]
            )
            self.tea_prob = tf.reduce_prod(self.tea_prob_all, axis = -1)

            # self.stu_prob = tf.cond(
            #     self.stu_prob_idx_fed,
            #     lambda: tf.gather_nd(self.prob, self.stu_prob_idx), 
            #     lambda: tf.constant([], dtype = tf.float32)
            # )

            self.bad_prob = tf.cond(
                self.stu_prob_idx_fed,
                lambda: tf.concat([self.tea_prob, tf.gather_nd(self.prob, self.stu_prob_idx)], axis = 0), 
                lambda: self.tea_prob
            )

            self.bad_log_prob = tf.log(self.bad_prob)
            self.adv = tf.placeholder(tf.float32, [None])

            if config.explore_method == 'anneal':
                self.loss = -1 * tf.reduce_mean(self.bad_log_prob * self.adv)
                self.train_op = tf.train.AdamOptimizer(config.lr).minimize(self.loss, global_step = self.global_step)
            elif config.explore_method == 'entropy':
                self.entropy = -1 * tf.reduce_sum(self.prob * tf.log(self.prob + 1e-12), axis = 1)
                self.loss = -1 * (tf.reduce_mean(self.bad_log_prob * self.adv, axis = 0) + config.entropy_param * tf.reduce_mean(self.entropy, axis = 0))
                self.train_op = tf.train.AdamOptimizer(config.lr).minimize(self.loss)
            


    def sample_action(self, menu, workplace, cf):
        if self.config.test or self.config.change_pair_test or self.config.order_free_test:
            action, prob_pre = self.sess.run(
                [self.argmax_action, self.prob_pre], 
                feed_dict = {
                    self.menu: menu,
                    self.workplace: workplace, 
                    self.cf: cf
                }
            )
        else:
            action, prob_pre = self.sess.run(
                [self.sampled_action, self.prob_pre], 
                feed_dict = {
                    self.menu: menu,
                    self.workplace: workplace, 
                    self.cf: cf
                }
            )

        return action, prob_pre


    def train_policy(self, menu, workplace, cf, tea_prob_idx, stu_prob_idx, stu_prob_idx_fed, adv):
        if self.config.explore_method == 'anneal':
            loss, epsilon, _ = self.sess.run(
                [self.loss, self.epsilon, self.train_op], 
                feed_dict = {
                    self.menu: menu,
                    self.workplace: workplace, 
                    self.cf: cf,
                    self.tea_prob_idx: tea_prob_idx,
                    self.stu_prob_idx: stu_prob_idx,
                    self.stu_prob_idx_fed: stu_prob_idx_fed,
                    self.adv: adv
                } if stu_prob_idx is not None else {
                    self.menu: menu,
                    self.workplace: workplace, 
                    self.cf: cf,
                    self.tea_prob_idx: tea_prob_idx,
                    self.stu_prob_idx_fed: stu_prob_idx_fed,
                    self.adv: adv
                }
            )
        elif self.config.explore_method == 'entropy':
            epsilon = np.nan
            loss, _ = self.sess.run(
                [self.loss, self.train_op], 
                feed_dict = {
                    self.menu: menu,
                    self.workplace: workplace, 
                    self.cf: cf,
                    self.tea_prob_idx: tea_prob_idx,
                    self.stu_prob_idx: stu_prob_idx,
                    self.stu_prob_idx_fed: stu_prob_idx_fed,
                    self.adv: adv
                } if stu_prob_idx is not None else {
                    self.menu: menu,
                    self.workplace: workplace, 
                    self.cf: cf,
                    self.tea_prob_idx: tea_prob_idx,
                    self.stu_prob_idx_fed: stu_prob_idx_fed,
                    self.adv: adv
                }
            )

        return loss, epsilon
