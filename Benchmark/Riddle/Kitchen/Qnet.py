import tensorflow as tf
import numpy as np
from gru import GRU
import os


class QNet:
    def __init__(self, config, sess, target=False):

        self.config = config
        self.sess = sess
        self.num_obv = self.config.num_obv
        self.num_action = self.config.num_action
        self.num_player = self.config.num_player
        self.num_ingredient = self.config.num_ingredient
        self.num_candidate = self.config.num_candidate
        self.hidden_dim = self.config.hidden_dim
        self.in_x_dim = self.config.in_x_dim
        self.out_x_dim = self.config.out_x_dim
        self.batch_size = self.config.batch_size
        self.max_time = self.config.max_time
        self.gamma = self.config.gamma

        if target:
            self.scope_name = 'target'
        else:
            self.scope_name = 'primary'

        with tf.variable_scope(self.scope_name):
            self.obv = tf.placeholder(tf.float32, [None, None, self.num_obv * self.num_ingredient])
            self.index = tf.placeholder(tf.int32, [None, None])
            self.index_one_hot = tf.one_hot(self.index, self.num_player)
            self.actions = tf.placeholder(tf.int32, [None, None, self.num_action])
            self.action_chosen = tf.placeholder(tf.int32, [None, None])

            # hidden state GRU
            self.gru = GRU(self.hidden_dim, self.in_x_dim, self.out_x_dim, config.gpu, self.sess)
            self.in_x = tf.placeholder(tf.float32, [None, None, self.in_x_dim])
            self.in_state = tf.placeholder(tf.float32, [None, self.hidden_dim])
            self.hidden_states = tf.placeholder(tf.float32, [None, None, self.out_x_dim])
            self.out_states = tf.placeholder(tf.float32, [None, self.hidden_dim])
            # fully connected layer
            # self.out_layers = [self.out_states]
            # for idx, out_fc_dim in enumerate(config.out_fc_dims):
            #     if idx == len(config.out_fc_dims) - 1:
            #         activation = None
            #     else:
            #         activation = tf.nn.leaky_relu
            #     self.out_layers.append(tf.layers.dense(self.out_layers[-1], out_fc_dim, activation=activation))
            # self.out_states_fc = self.out_layers[-1]

            if config.ad_hoc_structure:
                self.menu = tf.reshape(self.obv[:, :, :config.num_candidate * config.num_ingredient],
                    [tf.shape(self.obv)[0], tf.shape(self.obv)[1], config.num_candidate, config.num_ingredient])
                self.target = self.obv[:, :, config.num_candidate * config.num_ingredient: config.num_candidate * config.num_ingredient + config.num_candidate]
                self.prepared_ingred = self.obv[:, :, config.num_candidate * config.num_ingredient + config.num_candidate:]

                self.menu_context_vec = tf.reduce_sum(self.menu, axis=2, keepdims=True)
                self.menu_context = tf.concat([self.menu_context_vec] * config.num_candidate, axis=2)
                self.menu_with_context = tf.concat([self.menu, self.menu_context], axis=-1)
                self.menu_with_context_fc = tf.layers.dense(self.menu_with_context, config.dim_menu_with_context,
                                                            activation=tf.nn.leaky_relu)

                self.target_mat = tf.stack([self.target] * config.dim_menu_with_context, axis=-1)
                self.menu_attention = self.menu_with_context_fc * self.target_mat
                self.menu_attention_vec = tf.reduce_sum(self.menu_attention, axis=2)

                self.obv_ = tf.concat([self.menu_attention_vec, self.prepared_ingred], axis=-1)

            else:
                # self.obv_ = tf.concat([tf.reshape(self.obv, [-1, self.num_obv * self.num_ingredient]), \
                #     tf.reshape(self.hidden_states, [-1, self.out_x_dim])], -1)
                self.obv_ = tf.concat([self.obv, self.hidden_states], -1)

            # qnet
            self.input = tf.concat([self.obv_, self.index_one_hot], -1)
            self.layers = [self.input]
            for i, dim in enumerate(self.config.layer_dim):
                if i != len(self.config.layer_dim) - 1:
                    activation = tf.nn.leaky_relu
                else:
                    activation = None
                self.layers.append(tf.layers.dense(self.layers[-1], dim, activation=activation,
                                                   kernel_initializer=tf.initializers.he_normal()))
            self.q_values = self.layers[-1]

            self.target_q = tf.placeholder(tf.float32, [None, None])
            # self.action_chosen = tf.argmax(self.actions, axis=-1)
            # self.q_chosen = tf.gather_nd(self.q_values, tf.stack([tf.range(tf.shape(self.q_values)[0]),
            #                                                      self.action_chosen], 1))

            self.q_values_mask = tf.placeholder(tf.float32, [None, None, config.num_action])
            # self.loss = tf.reduce_sum(tf.square(self.target_q - self.q_chosen * self.target_q_mask))
            self.loss = tf.reduce_sum(tf.square(tf.reduce_sum(self.q_values * self.q_values_mask, axis=2)
                          - self.target_q)) / tf.reduce_sum(self.q_values_mask)
            # self.loss = tf.losses.mean_squared_error(self.target_q, self.q_chosen)
            self.train_opt = tf.train.AdamOptimizer(self.config.learning_rate).minimize(self.loss)
            # self.update_op = None

        self.q_primary_varlist_ = [v for v in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="primary")]
        self.q_target_varlist_ = [v for v in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="target")]
        self.q_target_update_ = tf.group([v_t.assign(v_t * (1 - 0.2) + v * 0.2) \
                                            for v_t, v in zip(self.q_target_varlist_, self.q_primary_varlist_)])

        self.total_loader_ = tf.train.Saver([v for v in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)])
        self.total_saver_ = tf.train.Saver([v for v in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)], max_to_keep=5)

    def save_ckpt(self, sess, ckpt_dir, global_step):
        if not os.path.exists(ckpt_dir):
            os.makedirs(ckpt_dir)
        if not os.path.exists(ckpt_dir):
            os.makedirs(ckpt_dir)

        self.total_saver_.save(sess, os.path.join(ckpt_dir, 'checkpoint'), global_step=global_step)
        print('Saved <%d> ckpt to %s' % (global_step, ckpt_dir))

    def restore_ckpt(self, sess, ckpt_dir):
        ckpt_status = tf.train.get_checkpoint_state(ckpt_dir)
        if ckpt_status:
            self.total_loader_.restore(sess, ckpt_status.model_checkpoint_path)
        if ckpt_status:
            print('Load model from %s' % ckpt_dir)
            return True
        print('Fail to load model from %s' % ckpt_dir)
        return False

    def get_hidden_states(self, sess, obv, action, in_state):

        # in_x = np.concatenate([np.reshape(obv, [-1, self.num_obv * self.num_ingredient]), action], axis=1)
        # in_x = np.expand_dims(in_x, axis=1)
        action = np.expand_dims(action, axis=1)
        in_x = np.concatenate([obv, action], -1)
        # in_x = np.expand_dims(in_x, axis=1)

        hidden_states, out_states = self.gru(in_x, in_state)

        # out_states = self.sess.run(self.out_states_fc, feed_dict={
        #     self.out_states: out_states
        # })

        return hidden_states, out_states

    def get_q(self, obv, hidden_states, index):

        q_values = self.sess.run(self.q_values, feed_dict={
            self.obv: obv,
            self.hidden_states: hidden_states,
            self.index: index
        })

        return q_values

    # def get_q_chosen(self, obv, hidden_states, player_index, action_chosen):
    #     # q_values = self.get_q(obv, hidden_states, player_index)
    #
    #     q_chosen = self.sess.run(self.q_chosen, feed_dict={
    #         self.obv: obv,
    #         self.hidden_states: hidden_states,
    #         self.index: player_index,
    #         self.action_chosen: action_chosen})
    #     return q_chosen

    def fit_target_q(self, obv, hidden_states, player_index, action_chosen, target_q, q_values_mask):
        train_opt, loss = self.sess.run([self.train_opt, self.loss], feed_dict={
            self.obv: obv,
            self.hidden_states: hidden_states,
            self.index: player_index,
            self.action_chosen: action_chosen,
            self.target_q: target_q,
            self.q_values_mask: q_values_mask
        })
        return loss

    # def target_net_update(self, q_var, target_q_var, soft=False):
    #     self.sess.run([v_t.assign(v) for v_t, v in zip(target_q_var, q_var)])

    # def set_target_net_update_op(self, q_primary_vars, q_target_vars, soft=False):
    #     if self.update_op is None:
    #         if soft:
    #             self.update_op = tf.group(
    #                 [v_t.assign(v_t * (1 - 0.2) + v * 0.2) for v_t, v in zip(q_target_vars, q_primary_vars)])
    #         else:
    #             self.update_op = tf.group([v_t.assign(v) for v_t, v in zip(q_target_vars, q_primary_vars)])
    #
    # def target_net_update(self):
    #     assert (self.update_op is not None)
    #     self.sess.run(self.update_op)

    def target_net_update(self):
        self.sess.run(self.q_target_update_)



