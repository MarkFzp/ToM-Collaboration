import tensorflow as tf

class QNet:
    def __init__(self, config, sess, target=False):
        self.config = config
        self.sess = sess
        self.num_player = config.num_player
        self.num_action = config.num_action
        if target:
            self.scope_name = 'target'
        else:
            self.scope_name = 'primary'
        with tf.variable_scope(self.scope_name):
            self.state = tf.placeholder(tf.float32, [None, config.q_state_dim])
            self.player_idx = tf.placeholder(tf.int32, [None])
            self.player_idx_one_hot = tf.one_hot(self.player_idx, self.num_player)
            
            if config.ad_hoc_structure:
                self.menu = tf.reshape(self.state[:, :config.num_candidate * config.num_ingredient], 
                    [-1, config.num_candidate, config.num_ingredient])
                self.target = self.state[:, config.num_candidate * config.num_ingredient: config.num_candidate * config.num_ingredient + config.num_candidate]
                self.prepared_ingred = self.state[:, config.num_candidate * config.num_ingredient + config.num_candidate:]
            
            # self.action_else = tf.placeholder(tf.int32, shape=(None, self.num_player))
            # self.action_else_one_hot = tf.one_hot(self.action_else, self.num_action)
            # self.action_else_one_hot = tf.reshape(self.action_else_one_hot, [-1, self.num_player * self.num_action])

            # self.player_idx = tf.placeholder(tf.int32, shape=(None,))
            # self.player_idx_one_hot = tf.one_hot(self.player_idx, self.num_player)

            if config.ad_hoc_structure:
                self.menu_context_vec = tf.reduce_sum(self.menu, axis = 1, keepdims = True)
                self.menu_context = tf.concat([self.menu_context_vec] * config.num_candidate, axis = 1)
                self.menu_with_context = tf.concat([self.menu, self.menu_context], axis = -1)
                self.menu_with_context_fc = tf.layers.dense(self.menu_with_context, config.dim_menu_with_context, activation = tf.nn.leaky_relu)

                self.target_mat = tf.stack([self.target] * config.dim_menu_with_context, axis = -1)
                self.menu_attention = self.menu_with_context_fc * self.target_mat
                self.menu_attention_vec = tf.reduce_sum(self.menu_attention, axis = 1)

                self.state_ = tf.concat([self.menu_attention_vec, self.prepared_ingred], axis = -1)
            
            else:
                self.state_ = self.state

            if config.q_use_idx:
                self.layers = [tf.concat([self.state_, self.player_idx_one_hot], axis = -1)]
            else:
                self.layers = [self.state_]

            for i, dim in enumerate(self.config.fc_layer_dims):
                if i != len(self.config.fc_layer_dims) - 1:
                    activation = tf.nn.leaky_relu
                elif config.q_tanh:
                    activation = tf.nn.tanh
                else:
                    activation = None
                self.layers.append(tf.layers.dense(self.layers[-1], dim, activation=activation, \
                    kernel_initializer=tf.initializers.he_normal()))
            self.q = self.layers[-1]

            assert(self.q.get_shape()[-1] == self.config.num_action)
            self.target_q = tf.placeholder(tf.float32, [None])
            self.action_self_chosen = tf.placeholder(tf.int32, [None])
            self.q_chosen = tf.gather_nd(self.q, tf.stack([tf.range(tf.shape(self.q)[0]), self.action_self_chosen], 1))
            self.loss = tf.losses.mean_squared_error(self.target_q, self.q_chosen)
            self.train_op = tf.train.AdamOptimizer(self.config.lr).minimize(self.loss)
            self.update_op = None

    

    def get_q(self, state, player_idx, action_self_chosen):
        q_chosen = self.sess.run(self.q_chosen, feed_dict = {
            self.state: state,
            self.player_idx: player_idx,
            self.action_self_chosen: action_self_chosen
        })
        return q_chosen


    
    def fit_target_q(self, state, player_idx, action_self_chosen, target_q):
        loss, q_chosen, _ = self.sess.run(
            [self.loss, self.q_chosen, self.train_op], 
            feed_dict = {
                self.state: state,
                self.player_idx: player_idx,
                self.action_self_chosen: action_self_chosen, 
                self.target_q: target_q
            }
        )
        return loss, q_chosen



    def set_target_net_update_op(self, q_primary_vars, q_target_vars, soft = False):
        if self.update_op is None:
            if soft:
                self.update_op = tf.group([v_t.assign(v_t * (1 - 0.2) + v * 0.2) for v_t, v in zip(q_target_vars, q_primary_vars)])
            else:
                self.update_op = tf.group([v_t.assign(v) for v_t, v in zip(q_target_vars, q_primary_vars)])
    

    
    def target_net_update(self):
        assert(self.update_op is not None)
        self.sess.run(self.update_op)
