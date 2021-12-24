import numpy as np

from Benchmark.MADDPG.mlp.policy import *

def p_func(inp, act_dim, scope, num_units, trainable=True, use_rnn=False, gpu=True):
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):

        # if use_rnn:
        #     in_state = tf.placeholder_with_default(tf.zeros([tf.shape(inp)[0], num_units]),
        #                                            [None, num_units])
        #     if gpu:
        #         cell = tf.keras.layers.CuDNNGRU(num_units, return_sequences = True, return_state = True)
        #     else:
        #         cell = tf.keras.layers.GRU(num_units, return_sequences = True, return_state = True)
        # else:
        L = tf.layers.dense(inp, num_units, activation=tf.nn.leaky_relu, trainable=trainable)
        L = tf.layers.dense(L, num_units, activation=tf.nn.leaky_relu, trainable=trainable)
        # L = tf.layers.dense(L, num_units, activation=tf.nn.leaky_relu, trainable=trainable)
        L = tf.layers.dense(L, act_dim, trainable=trainable)

    vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope)
    return L, vars

def q_func(inp, q_dim, scope, num_units, trainable=True):
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        L = tf.layers.dense(inp, num_units, activation=tf.nn.leaky_relu, trainable=trainable)
        L = tf.layers.dense(L, num_units, activation=tf.nn.leaky_relu, trainable=trainable)
        # L = tf.layers.dense(L, num_units, activation=tf.nn.leaky_relu, trainable=trainable)
        L = tf.layers.dense(L, q_dim, trainable=trainable)

    vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope)
    return tf.squeeze(L, axis=1), vars

def discount_with_dones(rewards, dones, gamma):
    discounted = []
    r = 0
    for reward, done in zip(rewards[::-1], dones[::-1]):
        r = reward + gamma*r
        r = r*(1.-done)
        discounted.append(r)
    return discounted[::-1]


def build_net(name, num_agents, index, num_dishes, num_ingredients, num_ensemble, act_dim, p_func, q_func, num_units, embed_dim, a_lr, c_lr, buffer_size, heuristic=False):

    '''
    placeholder
    '''
    menu_ph = tf.placeholder(tf.float32, [None, num_dishes, num_ingredients])
    wpeb_ph = tf.placeholder(tf.float32, [None, num_ingredients] )

    goal_ph = tf.placeholder(tf.int32, [None])

    temp_ph = tf.placeholder(tf.float32, [])

    if index == 0:
        bind = tf.range(tf.shape(goal_ph)[0])
        target_dish = tf.gather_nd(menu_ph, tf.stack([bind, goal_ph], axis=1))
        # embed = tf.layers.conv1d(tf.concat([target_dish[:,tf.newaxis,:], wpeb_ph[:,tf.newaxis,:]], axis=1), embed_dim, 1, use_bias=False)
        # embed = tf.concat([target_dish[:,tf.newaxis,:], wpeb_ph[:,tf.newaxis,:]], axis=1)
        # obs_ph = tf.reshape(embed, [-1,2*embed_dim])
        # obs_ph = tf.reshape(embed, [-1, 2 * num_ingredients])
        obs_ph = target_dish - wpeb_ph
    else:

        if heuristic:
            tmp = menu_ph - wpeb_ph[:,tf.newaxis,:]
            ind = tf.reduce_all(tmp >= 0, axis=-1, keepdims=True)
            embed = tmp * tf.cast(ind, tf.float32)
            obs_ph =  tf.reduce_sum(embed, axis=1)
        else:

            # embed = tf.layers.conv1d(tf.concat([menu_ph,wpeb_ph[:,tf.newaxis,:]], axis=1), embed_dim, 1,
            #                          use_bias=False)
            embed = tf.concat([menu_ph, wpeb_ph[:,tf.newaxis,:]], axis=1)
            # obs_ph = tf.reshape(embed, [-1, (num_dishes+1)*embed_dim])
            obs_ph = tf.reshape(embed, [-1, (num_dishes + 1) * num_ingredients])

    act_ph_n = [tf.placeholder(tf.int32, [None,]) for _ in range(num_agents)]

    target_q_ph = tf.placeholder(tf.float32, [None,])

    '''
    policy
    '''

    p_input = obs_ph
    policy = Policy(name, num_agents, index, num_ensemble, p_func, q_func, p_input, temp_ph, act_dim, num_units, a_lr, buffer_size, name+'_train_q')

    '''
    qnet
    '''

    ## for critic
    q_input = tf.concat([obs_ph] + [tf.one_hot(act_ph, act_dim) for act_ph in act_ph_n], 1)
    c_q, train_q_vars = q_func(q_input, 1, scope=name+"_train_q", num_units=num_units, trainable=True)

    c_loss = tf.reduce_mean(tf.square(c_q - target_q_ph))

    ## target
    target_q, target_q_vars = q_func(q_input, 1, scope=name+"_target_q", num_units=num_units, trainable=False)
    update_target_q = make_update_exp(train_q_vars, target_q_vars)
    '''
    critic train opt
    '''
    c_train_opt = tf.train.AdamOptimizer(c_lr).minimize(c_loss, var_list=train_q_vars)

    return {'ph': [menu_ph, wpeb_ph, goal_ph, act_ph_n, target_q_ph, temp_ph],
            'actor': policy,
            'critic': [c_q, target_q, c_loss, c_train_opt, update_target_q]}


class MADDPGAgentTrainer:
    def __init__(self, name, num_agents, agent_index, config, actor_model, critic_model):
        self.name = name
        self.n = num_agents
        self.agent_index = agent_index
        self.config = config

        # Create all the functions necessary to train the model
        dict = build_net(
            name,
            num_agents,
            agent_index,
            config.num_dishes,
            config.num_ingredients,
            config.num_ensemble,
            config.num_actions,
            actor_model,
            critic_model,
            config.dense,
            config.embed_dim,
            config.a_lr,
            config.c_lr,
            config.buffer_size
        )

        self.policy = dict['actor']
        self.train_q, self.target_q, self.c_loss, self.c_train_opt, self.update_target_q = dict['critic']

        self.menu_ph, self.wpeb_ph, self.goal_ph, self.act_ph_n, self.target_q_ph, self.temp_ph = dict['ph']

        self.random_policy_index = None

    def set_policy_index(self):
        ## called before each episode
        self.random_policy_index = np.random.randint(self.config.num_ensemble)

    def action(self, sess, menu, wpeb, goal, temp, train=True):
        assert self.random_policy_index is not None
        feed = {self.menu_ph: menu, self.wpeb_ph: wpeb, self.goal_ph: goal, self.temp_ph: temp}
        return sess.run([self.policy[self.random_policy_index]['train']['act_sample'],
                         self.policy[self.random_policy_index]['train']['entropy']] if train else
                        [self.policy[self.random_policy_index]['target']['act_sample'],
                        self.policy[self.random_policy_index]['target']['entropy']], feed_dict=feed)

    def experience(self, menu, workplace_embed, action, rewards, next_workplace_embed,  goal, terminal):
        # Store transition in the replay buffer.
        assert self.random_policy_index is not None
        self.policy[self.random_policy_index]['buffer']['buffer'].add(menu, workplace_embed, action, rewards, next_workplace_embed,  goal, terminal)

    def preupdate(self):
        self.policy[self.random_policy_index]['buffer']['replay_sample_index'] = None

    def update(self, sess, agents, temp, t):
        if not self.policy[self.random_policy_index]['buffer']['buffer'].is_available(): # replay buffer is not large enough
            return

        self.policy[self.random_policy_index]['buffer']['replay_sample_index'] = \
            self.policy[self.random_policy_index]['buffer']['buffer'].make_index(self.config.batch_size)
        # collect replay sample from all agents
        index = self.policy[self.random_policy_index]['buffer']['replay_sample_index']

        menu_n, wpeb_n, act_n, rewards_n, next_menu_n, next_wpeb_n, goal_n, terminal_n \
            = self.policy[self.random_policy_index]['buffer']['buffer'].sample_index(index)

        menu, wpeb, rewards, next_menu, next_wpeb, goal, terminal = \
            menu_n[self.agent_index], wpeb_n[self.agent_index], rewards_n[self.agent_index], \
            next_menu_n[self.agent_index], next_wpeb_n[self.agent_index], goal_n[self.agent_index], terminal_n[self.agent_index]

        # train q network
        num_sample = 1
        target_q = 0.0
        for i in range(num_sample):
            target_act_next_n = []
            for j, agent in enumerate(agents):
                raw_act_next = agent.action(sess, next_menu, next_wpeb, goal, temp, train=False)[0]
                null_act = np.full((next_menu.shape[0],), -1)
                mask = act_n[j] < 0
                target_act_next = raw_act_next * mask + (~mask) * null_act
                target_act_next_n.append(target_act_next)

            target_q_next = self.get_q(sess, next_menu, next_wpeb, target_act_next_n, goal, train=False)
            target_q += rewards + self.config.gamma * (1.0 - terminal) * target_q_next
        target_q /= num_sample
        feed_dict = dict(zip([self.menu_ph, self.wpeb_ph, self.goal_ph, self.target_q_ph]+self.act_ph_n,
                             [menu, wpeb, goal, target_q]+act_n))
        _, q_loss = sess.run([self.c_train_opt, self.c_loss], feed_dict=feed_dict)

        # train p network
        feed_dict = dict(zip([self.menu_ph, self.wpeb_ph, self.goal_ph, self.temp_ph],
                             [menu, wpeb, goal, temp]))
        _, p_loss  = sess.run([self.policy[self.random_policy_index]['opt']['train_opt'],
                              self.policy[self.random_policy_index]['opt']['loss']], feed_dict=feed_dict)

        if t % self.config.update_step == 0:
            self.p_update(sess)
            self.q_update(sess)

        return [q_loss, p_loss, np.mean(target_q), np.mean(rewards), np.mean(target_q_next), np.std(target_q)]

    def p_update(self, sess):
        sess.run([self.policy[i]['update'] for i in range(self.config.num_ensemble)])

    def q_update(self, sess):
        sess.run(self.update_target_q)

    def get_q(self, sess, menu, wpeb, act_n: list, goal, train=True):

        return sess.run(self.train_q if train else self.target_q,
                          feed_dict=dict(zip([self.menu_ph, self.wpeb_ph, self.goal_ph]+self.act_ph_n,
                                                [menu, wpeb, goal] + act_n)))


    def get_all_action_q(self, sess, menu, wpeb, goal, train=True):
        q = []
        bs = menu.shape[0]
        for i in range(self.config.num_ingredients):
            act = np.zeros((bs,))
            act.fill(i)
            act_n = [np.zeros_like(act) if k != self.agent_index else act for k in range(self.n)]
            q_ = self.get_q(sess, menu, wpeb, act_n, goal, train)
            q.append(q_)

        return np.stack(q, axis=1)
    def get_pi_logit(self, sess, menu, wpeb, goal, train=True):
        logit = sess.run(self.policy[self.random_policy_index]['train']['logits'] if train else
                         self.policy[self.random_policy_index]['target']['logits'], feed_dict={self.menu_ph:menu,
                                                                          self.wpeb_ph: wpeb,
                                                                          self.goal_ph: goal})
        return logit