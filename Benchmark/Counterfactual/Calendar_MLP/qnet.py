import tensorflow as tf

class QNet:
    def __init__(self, config, sess, target = False):
        self.config = config
        self.sess = sess

        if target:
            self.scope_name = 'target'
            self.update_op = None
        else:
            self.scope_name = 'primary'

        with tf.variable_scope(self.scope_name):
            self.p1_calendar = tf.placeholder(tf.float32, [None, config.num_slot])
            self.p2_calendar = tf.placeholder(tf.float32, [None, config.num_slot])
            self.prev_action_p_idx = tf.placeholder(tf.float32, [None, config.action_encode_dim + 2])
            # self.player_idx = tf.placeholder(tf.float32, [None, 2])
            
            self.layers = [tf.concat([self.p1_calendar, self.p2_calendar, self.prev_action_p_idx], axis = 1)]
            for i, dim in enumerate(self.config.fc_layer_dims):
                if i != len(self.config.fc_layer_dims) - 1:
                    activation = tf.nn.leaky_relu
                else:
                    activation = None
                self.layers.append(tf.layers.dense(self.layers[-1], dim, activation = activation))
            self.q = self.layers[-1]

            assert(self.q.get_shape()[-1] == self.config.num_action)
            self.target_q = tf.placeholder(tf.float32, [None])
            self.q_chosen_idx = tf.placeholder(tf.int32, [None, 2])
            self.q_chosen = tf.gather_nd(self.q, self.q_chosen_idx)
            self.q_pi_chosen_idx = tf.placeholder(tf.int32, [None, 1])
            self.q_pi_chosen = tf.gather_nd(self.q, self.q_pi_chosen_idx)

            self.loss = tf.losses.mean_squared_error(self.target_q, self.q_chosen)
            self.train_op = tf.train.AdamOptimizer(self.config.lr).minimize(self.loss)



    def get_q(self, p1_calendar, p2_calendar, prev_action_p_idx, q_chosen_idx):
        q_chosen = self.sess.run(
            self.q_chosen, 
            feed_dict = {
                self.p1_calendar: p1_calendar,
                self.p2_calendar: p2_calendar, 
                self.prev_action_p_idx: prev_action_p_idx, 
                self.q_chosen_idx: q_chosen_idx
            }
        )
        return q_chosen
    

    def get_all_q(self, p1_calendar, p2_calendar, prev_action_p_idx):
        q = self.sess.run(
            self.q, 
            feed_dict = {
                self.p1_calendar: p1_calendar,
                self.p2_calendar: p2_calendar, 
                self.prev_action_p_idx: prev_action_p_idx
            }
        )
        return q


    def get_q_pi(self, p1_calendar, p2_calendar, prev_action_p_idx, q_pi_chosen_idx):
        q_pi_chosen = self.sess.run(
            self.q_pi_chosen, 
            feed_dict = {
                self.p1_calendar: p1_calendar,
                self.p2_calendar: p2_calendar,
                self.prev_action_p_idx: prev_action_p_idx,
                self.q_pi_chosen: q_pi_chosen_idx
            }
        )
        return q_pi_chosen


    
    def fit_target_q(self, p1_calendar, p2_calendar, prev_action_p_idx, q_chosen_idx, target_q):
        loss, _ = self.sess.run(
            [self.loss, self.train_op], 
            feed_dict = {
                self.p1_calendar: p1_calendar,
                self.p2_calendar: p2_calendar, 
                self.prev_action_p_idx: prev_action_p_idx, 
                self.q_chosen_idx: q_chosen_idx,
                self.target_q: target_q
            }
        )
        return loss



    def set_target_net_update_op(self, q_primary_vars, q_target_vars, soft = False):
        if self.update_op is None:
            if soft:
                self.update_op = tf.group([v_t.assign(v_t * (1 - 0.2) + v * 0.2) for v_t, v in zip(q_target_vars, q_primary_vars)])
            else:
                self.update_op = tf.group([v_t.assign(v) for v_t, v in zip(q_target_vars, q_primary_vars)])
    

    
    def target_net_update(self):
        # assert(self.update_op is not None)
        self.sess.run(self.update_op)
