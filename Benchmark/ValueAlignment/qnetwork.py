import tensorflow as tf
import numpy as np


class QNetwork:
    def __init__(self, config):

        self.config = config
        self.num_goals = config.num_goals
        self.num_hactions = config.num_hactions
        self.num_ractions = config.num_ractions
        self.num_dishes = config.num_dishes
        self.num_ingredients = config.num_ingredients
        self.embed_dim = config.embed_dim

        self.temp_ph = tf.placeholder(tf.float32, [])
        self.menu_ph = tf.placeholder(tf.float32, [None, self.num_dishes, self.num_ingredients], name='menu_placeholder')
        self.workplace_embed_ph = tf.placeholder(tf.float32, [None, self.num_ingredients], name='workplace_embed')
        self.belief_ph = tf.placeholder(tf.float32, [None, self.num_goals], name='belief_placeholder')
        self.goal_ph = tf.placeholder(tf.int64, [None,], name='goal_placeholder')
        self.haction_ph = tf.placeholder(tf.float32, [None, self.num_hactions], name='human_action_placeholder')
        self.raction_ph = tf.placeholder(tf.float32, [None, self.num_ractions], name='robot_action_placeholder')
        self.next_haction_ph = tf.placeholder(tf.float32, [None, self.num_hactions], name='next_human_action_placeholder')
        self.next_raction_ph = tf.placeholder(tf.float32, [None, self.num_ractions], name='next_robot_action_placeholder')
        # self.target_q_ph = tf.placeholder(tf.float32, [None, self.num_hactions, self.num_goals], name='target_q_placeholder')
        self.target_q_ph = tf.placeholder(tf.float32, [None,],
                                          name='target_q_placeholder')

        self.q_per_ra_ph = tf.placeholder(tf.float32, [None, self.num_hactions], name='q_per_raction')

        self.global_step = tf.train.get_or_create_global_step()
        self.update_gs = tf.assign(self.global_step, self.global_step + 1) ## this is extremely annoying. screw tf

        self.train_q, self.train_qvars = self._build_qnet('train', True)
        self.target_q, self.target_qvars = self._build_qnet('target', False)

        self.hprob_per_ra = self._build_prob_opt(self.q_per_ra_ph)

        self.assign_opt = tf.group([target_v.assign(train_v) for target_v, train_v in zip(self.target_qvars, self.train_qvars)])

        self.loss, self.train_opt = self._build_train_opt()



    def _build_qnet(self, scope, trainable):
        with tf.variable_scope(scope):
            # embed = tf.layers.conv1d(tf.concat([self.menu_ph, self.workplace_embed_ph[:,tf.newaxis,:]], axis=1),
            #                          self.config.embed_dim, 1, 1, activation=tf.nn.leaky_relu, use_bias=False, trainable=trainable)
            # embed = tf.layers.conv1d(embed, self.config.embed_dim, 1, 1, activation=tf.nn.leaky_relu, use_bias=False, trainable=trainable)

            context = tf.reduce_sum(self.menu_ph-self.workplace_embed_ph[:,tf.newaxis,:], axis=1)
            bind = tf.cast(tf.range(tf.shape(self.goal_ph)[0]), tf.int64)
            target_dish = tf.gather_nd(self.menu_ph-self.workplace_embed_ph[:,tf.newaxis,:], tf.stack([bind, self.goal_ph], axis=1))
            belief_embed = tf.reduce_sum(self.menu_ph-self.workplace_embed_ph[:,tf.newaxis,:] * self.belief_ph[:,:,tf.newaxis], axis=1)
            embed = tf.concat([target_dish, belief_embed, context, self.raction_ph], axis=-1)
            L = tf.layers.dense(embed,
                                self.config.dense, activation=tf.nn.leaky_relu, trainable=trainable)
            L = tf.layers.dense(L, self.config.dense, activation=tf.nn.leaky_relu, trainable=trainable)
            L = tf.layers.dense(L, self.config.dense, activation=tf.nn.leaky_relu, trainable=trainable)
            # L = tf.layers.dense(L, self.config.dense, activation=tf.nn.leaky_relu, trainable=trainable)
            L = tf.layers.dense(L, self.num_hactions, trainable=trainable)

            # L = tf.reshape(L, [-1, self.num_hactions, self.num_goals])


        vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope)
        return L, vars

    def _build_train_opt(self):
        bind = tf.cast(tf.range(tf.shape(self.haction_ph)[0]), tf.int64)
        ha = tf.argmax(self.haction_ph, axis=-1)
        # nha = tf.argmax(self.next_haction_ph, axis=-1)

        # target_q = tf.gather_nd(self.target_q_ph, tf.stack([bind, nha, self.goal_ph], axis=1))
        train_q = tf.gather_nd(self.train_q, tf.stack([bind, ha], axis=1))
        target_q = self.target_q_ph

        self.svp = target_q
        self.out = train_q

        # self.loss_debug = tf.reduce_mean(tf.square(target_q - train_q))

        loss = tf.losses.mean_squared_error(target_q, train_q)


        train_opt = tf.train.AdamOptimizer(self.config.lr, beta1=self.config.lr_beta1, beta2=0.99,
                                              epsilon=self.config.lr_epsilon).minimize(loss, global_step=self.global_step,
                                                                                       var_list=self.train_qvars)

        return loss, train_opt

    def _build_prob_opt(self, q_ph):
        return tf.nn.softmax(q_ph / self.temp_ph, axis=1)

    def get_qvalues_and_hprob_matrix(self, sess, menu, wpeb, beliefs, temp, train=True):

        raeye = np.eye(self.config.num_ractions)
        qvalues, hprob = [], []
        bs = menu.shape[0]

        for i in range(self.config.num_ractions):
            raction = np.tile(raeye[i:i+1], [bs,1])
            qh, hp = self.get_q_all_hactions(sess, menu, wpeb, beliefs, raction, temp, train)

            qvalues.append(qh)
            hprob.append(hp)

        qvalues = np.stack(qvalues, axis=2)
        hprob = np.stack(hprob, axis=2)

        return qvalues, hprob


    def train(self, sess, robot, human, temp, menu, wpeb, beliefs, hactions, ractions,
              rewards, next_menu, next_wpeb, goal, terminal):

        _, hprob_sim_per_ra =\
            self.get_q_all_hactions(sess, menu, wpeb, beliefs, ractions, temp, train=False)
        hprob_per_ha_ra = hprob_sim_per_ra[np.arange(hactions.shape[0]),np.argmax(hactions,axis=-1),:]
        next_beliefs = robot.belief.fixed_point_belief(beliefs, hprob_per_ha_ra)

        next_q_all, next_hprob_all = self.get_qvalues_and_hprob_matrix(sess, next_menu, next_wpeb, next_beliefs, temp, train=False)
        next_ractions = robot.act(sess, next_q_all, next_hprob_all, next_beliefs, train=False)
        # next_q_per_ra, _ = \
        #     self.get_q_all_hactions(sess, next_menu, next_wpeb, next_beliefs, next_ractions, train=False)
        q_prime = next_q_all[np.arange(next_ractions.shape[0]),:,np.argmax(next_ractions,axis=-1),:]

        next_hactions = human.act(sess, q_prime, goal, temp, train=False)
        q = q_prime[np.arange(next_hactions.shape[0]),np.argmax(next_hactions, axis=-1),goal]
        #
        # add_discount = (next_wpeb - wpeb).sum(axis=-1)
        # add_discount[add_discount==2] = self.config.discount

        target_q = rewards + (self.config.discount ** 2) * q * (1- terminal)

        _, loss, train_q, out, svp = sess.run([self.train_opt, self.loss, self.train_q, self.out, self.svp], {self.menu_ph: menu,
                                                         self.workplace_embed_ph: wpeb,
                                                         self.belief_ph: beliefs,
                                                         self.raction_ph: ractions,
                                                         self.haction_ph: hactions,
                                                         self.target_q_ph: target_q,
                                                         self.goal_ph: goal})


        return loss, np.mean(rewards)


    def get_q_all_hactions(self, sess, menu, wpeb, rbelief, raction, temp, train=True):
        q_per_ra = []
        hp_per_ra = []
        for g in range(self.config.num_goals):
            goal = np.tile(np.array([g]), (menu.shape[0],))
            q, h = self.get_q_all_hactions_per_goal(sess, menu, wpeb, rbelief, raction, goal, temp, train)
            q_per_ra.append(q)
            hp_per_ra.append(h)

        return np.stack(q_per_ra,axis=2),np.stack(hp_per_ra,axis=2)

    def get_q_all_hactions_per_goal(self, sess, menu, wpeb, rbelief, raction, goal, temp, train=True):

        qvalues, _ = sess.run([self.train_q if train else self.target_q, self.update_gs], {self.menu_ph: menu,
                                    self.workplace_embed_ph: wpeb,
                                    self.belief_ph: rbelief,
                                    self.raction_ph: raction,
                                    self.goal_ph: goal,
                                    self.temp_ph: temp})

        hprob_per_ra = sess.run(self.hprob_per_ra, {self.q_per_ra_ph: qvalues, self.temp_ph:temp})

        return qvalues,hprob_per_ra

    def update_target_qnet(self, sess):
        sess.run(self.assign_opt)


class QNetworkv2:
    def __init__(self, config):

        self.config = config
        self.num_goals = config.num_goals
        self.num_hactions = config.num_hactions
        self.num_ractions = config.num_ractions
        self.num_dishes = config.num_dishes
        self.num_ingredients = config.num_ingredients
        self.embed_dim = config.embed_dim

        self.menu_ph = tf.placeholder(tf.float32, [None, self.num_dishes, self.num_ingredients], name='menu_placeholder')
        self.workplace_embed_ph = tf.placeholder(tf.float32, [None, self.num_ingredients], name='workplace_embed')
        self.belief_ph = tf.placeholder(tf.float32, [None, self.num_goals], name='belief_placeholder')
        self.goal_ph = tf.placeholder(tf.int64, [None,], name='goal_placeholder')
        self.haction_ph = tf.placeholder(tf.float32, [None, self.num_hactions], name='human_action_placeholder')
        self.raction_ph = tf.placeholder(tf.float32, [None, self.num_ractions], name='robot_action_placeholder')
        self.next_haction_ph = tf.placeholder(tf.float32, [None, self.num_hactions], name='next_human_action_placeholder')
        self.next_raction_ph = tf.placeholder(tf.float32, [None, self.num_ractions], name='next_robot_action_placeholder')
        self.target_q_ph = tf.placeholder(tf.float32, [None, self.num_hactions, self.num_ractions, self.num_goals], name='target_q_placeholder')

        self.global_step = tf.train.get_or_create_global_step()
        self.update_gs = tf.assign(self.global_step, self.global_step + 1) ## this is extremely annoying. screw tf

        self.train_q, self.train_qvars = self._build_qnet('train', True)
        self.target_q, self.target_qvars = self._build_qnet('target', False)

        self.hprob_train = self._build_prob_opt(self.train_q)
        self.hprob_target = self._build_prob_opt(self.target_q)

        self.assign_opt = tf.group([tf.assign(target_v, train_v) for target_v, train_v in zip(self.target_qvars, self.train_qvars)])

        self.loss, self.train_opt = self._build_train_opt()



    def _build_qnet(self, scope, trainable):
        with tf.variable_scope(scope):
            embed = tf.layers.conv1d(tf.concat([self.menu_ph, self.workplace_embed_ph[:,tf.newaxis,:]], axis=1),
                                     self.config.embed_dim, 1, 1, activation=tf.nn.leaky_relu, use_bias=False, trainable=trainable)
            embed = tf.layers.conv1d(embed, self.config.embed_dim, 1, 1, activation=tf.nn.leaky_relu, use_bias=False, trainable=trainable)

            belief_embed = tf.reduce_mean(embed[:,:-1,:] * self.belief_ph[:,:,tf.newaxis], axis=1)

            # embed = tf.concat([self.menu_ph, self.workplace_embed_ph[:,tf.newaxis,:]], axis=1)
            # belief_embed = self.belief_ph
            inp = tf.concat([tf.reshape(embed, [-1, (self.num_dishes+1)*self.num_ingredients]),
                             belief_embed], axis=1)
            L = tf.layers.dense(inp,
                                self.config.dense, activation=tf.nn.leaky_relu, trainable=trainable)
            L = tf.layers.dense(L, self.config.dense, activation=tf.nn.leaky_relu, trainable=trainable)
            L = tf.layers.dense(L, self.config.dense, activation=tf.nn.leaky_relu, trainable=trainable)
            # L = tf.layers.dense(L, self.config.dense, activation=tf.nn.leaky_relu, trainable=trainable)
            L = tf.layers.dense(L, self.num_hactions*self.num_ractions*self.num_goals, trainable=trainable)

            L = tf.reshape(L, [-1, self.num_hactions, self.num_ractions, self.num_goals])

        vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope)
        return L, vars

    def _build_train_opt(self):
        bind = tf.cast(tf.range(tf.shape(self.haction_ph)[0]), tf.int64)
        ha = tf.argmax(self.haction_ph, axis=-1)
        ra = tf.argmax(self.raction_ph, axis=-1)
        nha = tf.argmax(self.next_haction_ph, axis=-1)
        nra = tf.argmax(self.next_raction_ph, axis=-1)

        target_q = tf.gather_nd(self.target_q_ph, tf.stack([bind, nha, nra, self.goal_ph], axis=1))
        train_q = tf.gather_nd(self.train_q, tf.stack([bind, ha, ra, self.goal_ph], axis=1))

        # self.loss_debug = tf.reduce_mean(tf.square(target_q - train_q))

        loss = tf.losses.mean_squared_error(target_q, train_q)


        train_opt = tf.train.AdamOptimizer(self.config.lr, beta1=self.config.lr_beta1, beta2=0.999,
                                              epsilon=self.config.lr_epsilon).minimize(loss, global_step=self.global_step,
                                                                                       var_list=self.train_qvars)

        return loss, train_opt

    def _build_prob_opt(self, q_ph):
        return tf.nn.softmax(self.config.beta * q_ph, axis=1)

    def get_qvalues_and_hprob_matrix(self, sess, menu, wpeb, beliefs, train=True):

        return self.get_q(sess, menu, wpeb, beliefs, train)


    def train(self, sess, robot, human, menu, wpeb, beliefs, hactions, ractions,
              rewards, next_menu, next_wpeb, goal, terminal):

        _, hprob =\
            self.get_q(sess, menu, wpeb, beliefs, train=False)
        hprob_per_ha_ra = hprob[np.arange(hactions.shape[0]),np.argmax(hactions,axis=-1),np.argmax(ractions,axis=-1),:]
        next_beliefs = robot.belief.fixed_point_belief(beliefs, hprob_per_ha_ra)

        q_prime, next_hprob_all = self.get_q(sess, next_menu, next_wpeb, next_beliefs, train=False)
        next_ractions = robot.act(sess, q_prime, next_hprob_all, next_beliefs, train=False)
        # next_q_per_ra, _ = \
        #     self.get_q_all_hactions(sess, next_menu, next_wpeb, next_beliefs, next_ractions, train=False)
        q = q_prime[np.arange(next_ractions.shape[0]),:,np.argmax(next_ractions,axis=-1),:]
        next_hactions = human.act(sess, q, goal, self.config.beta, False)

        target_q = rewards[:,np.newaxis,np.newaxis,np.newaxis] + self.config.discount * q_prime * (1- terminal[:,np.newaxis,np.newaxis,np.newaxis])

        _, loss, train_q = sess.run([self.train_opt, self.loss, self.train_q], {self.menu_ph: menu,
                                                         self.workplace_embed_ph: wpeb,
                                                         self.belief_ph: beliefs,
                                                         self.haction_ph: hactions,
                                                         self.raction_ph: ractions,
                                                         self.next_haction_ph: next_hactions,
                                                         self.next_raction_ph: next_ractions,
                                                         self.target_q_ph: target_q,
                                                         self.goal_ph: goal})


        return loss


    def get_q(self, sess, menu, wpeb, rbelief, train=True):
        '''

        :return: q_target: [bs, goal], qvalues: [bs, num_hactions, goal],
                hprob_per_ha_ra: [bs, goal], hprob_per_ra: [bs, hactions, goal]
        '''
        q_train, hprob, _ = sess.run([self.train_q if train else self.target_q,
                              self.hprob_train if train else self.hprob_target,
                              self.update_gs], {self.menu_ph: menu,
                                    self.workplace_embed_ph: wpeb,
                                    self.belief_ph: rbelief})

        return q_train, hprob


    def update_target_qnet(self, sess):
        sess.run(self.assign_opt)
