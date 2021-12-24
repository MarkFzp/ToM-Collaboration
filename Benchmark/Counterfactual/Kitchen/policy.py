import tensorflow as tf
import numpy as np
import pdb

class Policy:
    def __init__(self, config, sess, player = None):
        self.sess = sess
        self.config = config
        self.hidden_dim = self.config.hidden_dim
        self.num_action = self.config.num_action
        self.num_player = self.config.num_player
        # menu, target_in_mind, prepared_ingredients, prev_action, player_idx
        self.in_x_dim = self.config.in_x_dim
        self.gpu = self.config.gpu

        self.one_hot_action_helper = np.eye(self.num_action)
        self.null_action_one_hot = np.zeros([self.config.num_action])
        self.one_hot_action_helper = np.concatenate([self.one_hot_action_helper, np.expand_dims(self.null_action_one_hot, axis = 0)], axis = 0)
        self.one_hot_action = lambda x: self.one_hot_action_helper[x]

        self.one_hot_p_idx_helper = np.eye(self.num_player)
        self.one_hot_p_idx = lambda x: self.one_hot_p_idx_helper[x]

        self.expand_batch_time = lambda x: np.expand_dims(np.expand_dims(x, 0), 0)

        with tf.variable_scope('policy_{}'.format(player) if player is not None else 'policy'):
            # batch_size * seq_len * in_x_dim
            self.in_x = tf.placeholder(tf.float32, [None, None, config.in_x_dim])
            self.in_state = tf.placeholder_with_default(tf.zeros([tf.shape(self.in_x)[0], self.hidden_dim]), [None, self.hidden_dim])

            if config.ad_hoc_structure:
                self.menu = tf.reshape(self.in_x[:, :, :config.num_candidate * config.num_ingredient], 
                    [tf.shape(self.in_x)[0], tf.shape(self.in_x)[1], config.num_candidate, config.num_ingredient])
                self.target = self.in_x[:, :, config.num_candidate * config.num_ingredient: \
                    config.num_candidate * config.num_ingredient + config.num_candidate]
                self.ing_act_idx = self.in_x[:, :, config.num_candidate * config.num_ingredient + config.num_candidate:]
                
                self.menu_context_vec = tf.reduce_sum(self.menu, axis = 2, keepdims = True)
                self.menu_context = tf.concat([self.menu_context_vec] * config.num_candidate, axis = 2)
                self.menu_with_context = tf.concat([self.menu, self.menu_context], axis = -1)
                self.menu_with_context_fc = tf.layers.dense(self.menu_with_context, config.dim_menu_with_context, activation = tf.nn.leaky_relu)

                self.target_mat = tf.stack([self.target] * config.dim_menu_with_context, axis = -1)
                self.menu_attention = self.menu_with_context_fc * self.target_mat
                self.menu_attention_vec = tf.reduce_sum(self.menu_attention, axis = 2)

                self.in_x_ = tf.concat([self.menu_attention_vec, self.ing_act_idx], axis = -1)
            
            else:
                self.in_x_ = self.in_x
            
            self.in_layers = [self.in_x_]
            for in_fc_dim in config.in_fc_dims:
                self.in_layers.append(tf.layers.dense(self.in_layers[-1], in_fc_dim, activation = tf.nn.leaky_relu))
            self.in_x_fc = self.in_layers[-1]

            # if self.gpu:
            #     self.cell = tf.keras.layers.CuDNNGRU(self.hidden_dim, return_sequences = True, return_state = True)
            # else:
            self.cell = tf.keras.layers.GRU(self.hidden_dim, return_sequences = True, return_state = True, name = 'gru' if player is None else 'gru_1')
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

            self.action_prob_pre_anneal = tf.nn.softmax(self.out_logit, axis = -1)
            
            if config.explore_method == 'anneal':
                self.global_step = tf.Variable(0, name = 'global_step', trainable = False)
                self.epsilon = tf.train.polynomial_decay(
                    config.init_epsilon, 
                    self.global_step, 
                    config.linear_decay_step, 
                    config.end_epsilon, 
                    power = 1.0, 
                    name = 'epsilon'
                )
                self.uniform_prob = tf.constant(np.ones((self.num_action,)) / self.num_action, dtype = tf.float32)
                self.action_prob = (1 - self.epsilon) * self.action_prob_pre_anneal + self.epsilon * self.uniform_prob
            elif config.explore_method == 'entropy':
                self.action_prob = self.action_prob_pre_anneal
            else:
                raise Exception()

            self.log_prob_idx = tf.placeholder(tf.int32, [None, 3])
            self.action_prob_selected = tf.gather_nd(self.action_prob, self.log_prob_idx)
            self.adv = tf.placeholder(tf.float32, [None])

            if config.explore_method == 'anneal':
                self.loss = -1 * tf.reduce_sum(tf.log(self.action_prob_selected) * self.adv) / (tf.cast(tf.shape(self.in_x_)[0], tf.float32) / 2)
                self.train_op = tf.train.AdamOptimizer(config.lr).minimize(self.loss, global_step = self.global_step)
            elif config.explore_method == 'entropy':
                self.entropy_prob_idx = self.log_prob_idx[:, :2]
                self.entropy_prob_selected = tf.gather_nd(self.action_prob, self.entropy_prob_idx)
                self.entropy = -1 * tf.reduce_sum(self.entropy_prob_selected * tf.log(self.entropy_prob_selected + 1e-12), axis = 1)
                self.loss = -1 * (tf.reduce_mean(tf.log(self.action_prob_selected) * self.adv) + config.entropy_param * tf.reduce_mean(self.entropy, axis = 0))
                self.train_op = tf.train.AdamOptimizer(config.lr).minimize(self.loss)



    # def sample_act(self, obs, prev_action, player_idx, in_state, epsilon):
    #     batch_size = obs.shape[0]
    #     prev_action_one_hot = self.one_hot_action(prev_action)
    #     player_idx_one_hot = self.one_hot_p_idx(player_idx)
        
    #     in_x = np.concatenate([obs, prev_action_one_hot, player_idx_one_hot], axis = 1)
    #     in_x = np.expand_dims(in_x, axis = 1)
        
    #     out_seq, out_state = self.gru(in_x, in_state)
    #     out_seq = np.squeeze(out_seq, axis = 1).astype(np.float64)
        
    #     exp_out_seq = np.exp(out_seq)
    #     action_prob = (1 - epsilon) * exp_out_seq / (np.sum(exp_out_seq, axis = 1) + 1e-10) + \
    #                 epsilon * np.ones([batch_size, self.num_action]) / self.num_action
    #     action = []
    #     for prob in action_prob:
    #         action.append(np.random.choice(self.num_action, p = prob))
    #     action = np.array(action)
        
    #     return action, action_prob, out_state




    def single_sample_act(self, obs, prev_action, player_idx, in_state):
        prev_action_one_hot = self.one_hot_action(prev_action)
        player_idx_one_hot = self.one_hot_p_idx(player_idx)
        in_x = np.concatenate([obs, prev_action_one_hot, player_idx_one_hot], axis = 0)
        in_x = np.expand_dims(in_x, axis = 0)
        in_x = np.expand_dims(in_x, axis = 0)
        
        action_prob, out_state = self.sess.run(
            [self.action_prob, self.out_state], 
            feed_dict = {
                self.in_x: in_x,
                self.in_state: np.expand_dims(in_state, 0)
            } if in_state is not None else {
                self.in_x: in_x
            }
        )

        action_prob = action_prob[0][0]
        out_state = out_state[0]
        if self.config.test or self.config.change_pair_test:
            action = np.argmax(action_prob)
        else:
            action = np.random.choice(self.num_action, p = action_prob)

        return action, action_prob, out_state

    
    

    def train(self, in_x, log_prob_idx, adv):
        if self.config.explore_method == 'anneal':
            loss, eps, _ = self.sess.run(
                [self.loss, self.epsilon, self.train_op], 
                feed_dict = {
                    self.in_x: in_x,
                    self.log_prob_idx: log_prob_idx,
                    self.adv: adv
                }
            )
        elif self.config.explore_method == 'entropy':
            loss, _ = self.sess.run(
                [self.loss, self.train_op], 
                feed_dict = {
                    self.in_x: in_x,
                    self.log_prob_idx: log_prob_idx,
                    self.adv: adv
                }
            )
            eps = np.nan
        
        return loss, eps
