import tensorflow as tf
import numpy as np


class QNet_const_mess:
    def __init__(self, config, sess, qnet_name_appendix, target = False):
        self.sess = sess
        self.config = config

        if target:
            self.variable_scope_name = 'QNet_target'
            self.update_op = None
        else:
            self.variable_scope_name = 'QNet_primary'

        if qnet_name_appendix is not None:
            self.variable_scope_name += '_{}'.format(qnet_name_appendix)

        with tf.variable_scope(self.variable_scope_name):
            self.noise = tf.random.normal([config.batch_size, config.max_attribute, config.message_dim], mean = 0.0, stddev = float(config.noise_std))
            self.global_step = tf.Variable(0, name = 'global_step', trainable = False)
            self.epsilon = tf.train.polynomial_decay(
                config.init_eps, 
                self.global_step, 
                config.linear_decay_step, 
                config.end_eps, 
                power = 1.0, 
                name = 'epsilon'
            )
            self.epsilon_1d = tf.reshape(self.epsilon, [-1])
            self.unif_rand = tf.random.uniform([config.batch_size, config.max_attribute], minval = 0.0, maxval = 1.0)

            # time invariant for a traj
            self.menu = tf.placeholder(tf.float32, [config.batch_size, config.num_candidate, config.num_ingredient])
            if config.in_arch == 'mlp':
                self.menu_flat = tf.reshape(self.menu, [config.batch_size, config.num_candidate * config.num_ingredient])
            

            self.workplaces = [tf.zeros([config.batch_size, config.num_ingredient])]
            self.p1_target = tf.placeholder(tf.int32, [config.batch_size])
            self.p1_target_oh = tf.one_hot(self.p1_target, depth = config.num_candidate)
            self.p2_target_oh = tf.zeros([config.batch_size, config.num_candidate])

            self.p1_prev_actions = [tf.fill([config.batch_size], -1)]
            self.p1_prev_action_ohs = [tf.one_hot(self.p1_prev_actions[-1], config.num_action)]
            self.p2_prev_actions = [tf.fill([config.batch_size], -1)]
            self.p2_prev_action_ohs = [tf.one_hot(self.p2_prev_actions[-1], config.num_action)]
            self.actions = []
            self.action_fed = tf.placeholder_with_default(tf.fill([config.batch_size, config.max_attribute], -1), [config.batch_size, config.max_attribute])
            self.is_action_fed = tf.placeholder(tf.bool, shape = [])

            self.p1_idx = tf.one_hot(tf.zeros([config.batch_size], dtype = tf.int32), depth = 2)
            self.p2_idx = tf.one_hot(tf.ones([config.batch_size], dtype = tf.int32), depth = 2)

            if config.message_activation == 'sigmoid':
                self.default_message = tf.fill([config.batch_size, config.message_dim], 0.5)
                self.messages = [self.default_message]
                self.message_activation = tf.sigmoid
            elif config.message_activation == 'tanh':
                self.default_message = tf.zeros([config.batch_size, config.message_dim])
                self.messages = [self.default_message]
                self.message_activation = tf.tanh

            self.p1_hs = [[tf.zeros([config.batch_size, hidden_dim]) for hidden_dim in config.hidden_dims]]
            self.p2_hs = [[tf.zeros([config.batch_size, hidden_dim]) for hidden_dim in config.hidden_dims]]
            self.grus = [tf.nn.rnn_cell.GRUCell(hidden_dim) for hidden_dim in config.hidden_dims]
            self.gru_layer_num = len(config.hidden_dims)

            self.action_uniform_dis = tf.distributions.Categorical(probs = tf.ones([config.num_action]) / config.num_action)

            if config.in_arch == 'mlp':
                self.in_layers = [tf.layers.Dense(fc_dim, activation = tf.nn.leaky_relu) for fc_dim in config.in_fc_dims]
            if config.out_arch == 'mlp':
                self.out_layers = [
                    tf.layers.Dense(fc_dim, activation = tf.nn.leaky_relu if i != len(config.out_fc_dims) - 1 else None) 
                    for i, fc_dim in enumerate(config.out_fc_dims)
                ]
            self.qs = []

            for t in range(config.max_attribute):
                workplace = self.workplaces[-1]
                if t % 2 == 0:
                    target = self.p1_target_oh
                    player_idx = self.p1_idx
                    hiddens = self.p1_hs[-1]
                else:
                    target = self.p2_target_oh
                    player_idx = self.p2_idx
                    hiddens = self.p2_hs[-1]
                p1_prev_action_oh = self.p1_prev_action_ohs[-1]
                p2_prev_action_oh = self.p2_prev_action_ohs[-1]
                
                # if config.constant_message_test:
                message = self.default_message
                # else:
                #     message = self.messages[-1]
                
                if config.in_arch == 'mlp':
                    in_raw = tf.concat([self.menu_flat, workplace, target, player_idx, p1_prev_action_oh, p2_prev_action_oh, message], axis = 1)
                    inter_layers = [in_raw]
                    for layer_obj in self.in_layers:
                        inter_layers.append(layer_obj(inter_layers[-1]))
                    in_x = inter_layers[-1]
                
                inter_layers = [in_x]
                new_hiddens = []
                for depth, gru in enumerate(self.grus):
                    out, _ = gru(inter_layers[-1], hiddens[depth])
                    new_hiddens.append(out)
                    inter_layers.append(out)
                out_x = inter_layers[-1]

                if t % 2 == 0:
                    self.p1_hs.append(new_hiddens)
                else:
                    self.p2_hs.append(new_hiddens)

                if config.out_arch == 'mlp':
                    inter_layers = [out_x]
                    for layer_obj in self.out_layers:
                        inter_layers.append(layer_obj(inter_layers[-1]))
                
                q_and_m = inter_layers[-1]
                assert(q_and_m.get_shape()[-1] == config.num_action + config.message_dim)

                q = q_and_m[:, :config.num_action]
                message = self.message_activation(q_and_m[:, config.num_action: ] + self.noise[:, t, :])

                self.qs.append(q)
                self.messages.append(message)

                action = tf.cast(tf.argmax(q, axis = 1), tf.int32)
                
                action_oh = tf.one_hot(action, config.num_action)

                if t % 2 == 0:
                    self.p1_prev_actions.append(action)
                    self.p1_prev_action_ohs.append(action_oh)
                else:
                    self.p2_prev_actions.append(action)
                    self.p2_prev_action_ohs.append(action_oh)
                self.actions.append(action)
                
                new_workplace = workplace + action_oh
                self.workplaces.append(new_workplace)
            
            self.action_tensor = tf.stack(self.actions, axis = 1)
            
            self.q_tensor = tf.stack(self.qs, axis = 1)
            self.q_max = tf.reduce_max(self.q_tensor, axis = 2)
            self.q_spvs = tf.placeholder(tf.float32, [None])
            self.q_idx = tf.placeholder(tf.int32, [None, 3])
            self.q_chosen = tf.gather_nd(self.q_tensor, self.q_idx)

            self.loss = tf.losses.mean_squared_error(self.q_spvs, self.q_chosen)
            self.train_op = tf.train.AdamOptimizer(self.config.lr).minimize(self.loss, global_step = self.global_step)


    def get_action(self, menu, p1_target):
        action = self.sess.run(
            self.action_tensor,
            feed_dict = {
                self.menu: menu,
                self.p1_target: p1_target
            }
        )

        return action


    def set_update_weights_op(self, q_primary_vars, q_constant_mess_vars):
        if self.update_op is None:
            self.update_op = tf.group([v_t.assign(v) for v_t, v in zip(q_constant_mess_vars, q_primary_vars)])


    def update_weights(self):
        self.sess.run(self.update_op)
