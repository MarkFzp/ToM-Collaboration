import tensorflow as tf

import numpy as np

from Benchmark.MADDPGC.rnn_gumbel_rnnq.policy import *

def p_func(inp, in_state, calendar_tensor, act_dim, scope, num_units, trainable=True, gpu=True):
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):

        if gpu:
            cell = tf.keras.layers.CuDNNGRU(num_units, return_sequences = True, return_state = True, trainable=trainable)
        else:
            cell = tf.keras.layers.GRU(num_units, return_sequences = True, return_state = True, trainable=trainable)

        out_seq, out_state = cell(inp, initial_state=in_state)

        # attention = tf.nn.softmax(out_seq)
        # attended_calendar = tf.reduce_sum(calendar_tensor * attention[:,:,:,tf.newaxis], axis=-2)
        # L = tf.concat([attended_calendar, inp[:,:,calendar_tensor.get_shape().as_list()[-1]:]], axis=-1)

        L = tf.layers.dense(out_seq, num_units, activation=tf.nn.leaky_relu, trainable=trainable)
        L = tf.layers.dense(L, num_units, activation=tf.nn.leaky_relu, trainable=trainable)
        # L = tf.layers.dense(L, num_units, activation=tf.nn.leaky_relu, trainable=trainable)
        # L = tf.layers.dense(L, num_units, activation=tf.nn.leaky_relu, trainable=trainable)
        L = tf.layers.dense(L, act_dim, trainable=trainable)

    vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope)
    return [L, out_state], vars

def q_func(inp, in_state, calendar_tensor, q_dim, scope, num_units, trainable=True, gpu=True):
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):

        if gpu:
            cell = tf.keras.layers.CuDNNGRU(num_units, return_sequences = True, return_state = True, trainable=trainable)
        else:
            cell = tf.keras.layers.GRU(num_units, return_sequences = True, return_state = True, trainable=trainable)

        out_seq, out_state = cell(inp, initial_state=in_state)
        # attention = tf.nn.softmax(out_seq)
        # attended_calendar = tf.reduce_sum(calendar_tensor * attention[:, :, :, tf.newaxis], axis=-2)
        # L = tf.concat([attended_calendar, inp[:, :, calendar_tensor.get_shape().as_list()[-1]:]], axis=-1)

        L = tf.layers.dense(out_seq, num_units, activation=tf.nn.leaky_relu, trainable=trainable)
        L = tf.layers.dense(L, num_units, activation=tf.nn.leaky_relu, trainable=trainable)
        L = tf.layers.dense(L, num_units, activation=tf.nn.leaky_relu, trainable=trainable)
        # L = tf.layers.dense(L, num_units, activation=tf.nn.leaky_relu, trainable=trainable)
        # L = tf.layers.dense(L, num_units, activation=tf.nn.leaky_relu, trainable=trainable)
        L = tf.layers.dense(L, q_dim, trainable=trainable)

    vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope)
    return [tf.squeeze(L, axis=-1), out_state], vars



def build_net(name, num_agents, index, calendar_tensor_,action_tensor_, num_slots, num_ensemble, act_dim, p_func, q_func, num_units,
              embed_dim, a_lr, c_lr, init_global_step, lr_decay_step, lr_decay_rate, beta, buffer_size, max_traj_len, use_gpu):

    '''
    placeholder
    '''
    private_ph = tf.placeholder(tf.float32, [None, None, num_slots])

    temp_ph = tf.placeholder(tf.float32, [])


    mask_ph = tf.placeholder(tf.bool, [None, None]) ## ~terminal is not enough, alternating null actions are also important

    calendar_tensor = tf.tile(tf.constant(calendar_tensor_[1:], tf.float32)[tf.newaxis,  tf.newaxis],
                              (tf.shape(private_ph)[0], tf.shape(private_ph)[1], 1, 1))  ## ignore the first calendar slot?
    # calendar_encoding = tf.layers.dense(calendar_tensor, calendar_tensor.shape[-1], activation=tf.nn.leaky_relu)
    # calendar_encoding = tf.layers.dense(calendar_encoding, calendar_tensor.shape[-1], activation=tf.nn.leaky_relu)
    # calendar_encoding = tf.reduce_mean(calendar_tensor, axis=-2)
    # calendar_encoding = tf.tile(calendar_encoding, (tf.shape(private_ph)[0], tf.shape(private_ph)[1], 1))
    action_tensor = tf.constant(action_tensor_, tf.float32)

    act_ph_n_q = [tf.placeholder(tf.int32, [None, None]) for _ in range(num_agents)]
    last_act_ph_n_policy = [tf.placeholder(tf.int32, [None, None]) for _ in range(num_agents)]
    last_act_ph_n_policy_ = [tf.gather(action_tensor, act) for act in last_act_ph_n_policy]

    # obs_ph_policy = calendar_tensor
    obs_ph = tf.concat([tf.tile(private_ph, [1,1,2]) - lact for lact in last_act_ph_n_policy_], axis=-1)

    in_state_ph = tf.placeholder_with_default(tf.zeros([tf.shape(private_ph)[0], num_units]),
                                              [None, num_units])
    q_in_state_ph = tf.placeholder_with_default(tf.zeros([tf.shape(private_ph)[0], num_units]),
                                                [None, num_units])

    target_q_ph = tf.placeholder(tf.float32, [None,None])

    '''
    policy
    '''
    inp = {'calendar_tensor':calendar_tensor, 'action_tensor':action_tensor, 'p_input':obs_ph, 'in_state':in_state_ph, 'private_ph':private_ph,
           'q_in_state_ph':q_in_state_ph, 'mask':mask_ph, 'temp':temp_ph, 'act_ph_n_q':act_ph_n_q, 'target_q_ph':target_q_ph}
    policy = Policy(name, num_agents, index, num_ensemble, p_func, q_func,inp, act_dim, num_units, a_lr,
                    init_global_step, lr_decay_step, lr_decay_rate, beta,
                    buffer_size, max_traj_len, name+'_train_q', name+'_target_q', use_gpu)
    '''
    qnet
    '''

    ## for critic
    q_input = tf.concat([obs_ph] + [tf.tile(private_ph, [1,1,2]) - tf.gather(action_tensor, act) for act in act_ph_n_q], axis=-1)
    [c_q, q_out_state], train_q_vars = q_func(q_input, q_in_state_ph, calendar_tensor,1, scope=name+"_train_q", num_units=num_units, trainable=True, gpu=use_gpu)

    c_loss = tf.reduce_mean(tf.square(tf.boolean_mask(c_q, mask_ph) - tf.boolean_mask(target_q_ph, mask_ph)))

    ## target
    [target_q, target_q_out_state], target_q_vars = q_func(q_input, q_in_state_ph, calendar_tensor, 1,  scope=name+"_target_q", num_units=num_units, trainable=False, gpu=use_gpu)
    update_target_q = make_update_exp(train_q_vars, target_q_vars)
    '''
    critic train opt
    '''
    c_train_opt = tf.train.AdamOptimizer(c_lr).minimize(c_loss, var_list=train_q_vars)

    return {'ph': [private_ph, last_act_ph_n_policy, act_ph_n_q, target_q_ph, temp_ph, in_state_ph, mask_ph],
            'actor': policy,
            'critic': [c_q, q_out_state, target_q_out_state, target_q, c_loss, c_train_opt, update_target_q, train_q_vars, target_q_vars]}


class MADDPGAgentTrainer:
    def __init__(self, name, num_agents, agent_index, config, actor_model, critic_model, calendar_tensor, action_tensor):
        self.name = name
        self.n = num_agents
        self.agent_index = agent_index
        self.config = config

        # Create all the functions necessary to train the model
        dict = build_net(
            name,
            num_agents,
            agent_index,
            calendar_tensor,
            action_tensor,
            config.num_slots,
            config.num_ensemble,
            config.num_actions,
            actor_model,
            critic_model,
            config.dense,
            config.embed_dim,
            config.a_lr,
            config.c_lr,
            config.init_global_step,
            config.lr_decay_step,
            config.lr_decay_rate,
            config.beta,
            config.buffer_size,
            config.max_traj_len,
            config.use_gpu
        )

        self.policy = dict['actor']
        self.train_q, self.q_out_state, self.target_q_out_state, self.target_q, self.c_loss, self.c_train_opt, self.update_target_q, self.train_q_vars, self.target_q_vars = dict['critic']

        self.private_ph, self.last_act_ph_n_policy, self.act_ph_n_q, self.target_q_ph, self.temp_ph, self.in_state_ph, self.mask_ph = dict['ph']

        self.random_policy_index = None

    def set_policy_index(self):
        ## called before each episode
        self.random_policy_index = np.random.randint(self.config.num_ensemble)

    def action(self, sess, private, last_action_n:list, temp, mask=np.array([[True]]), in_state=None,  train=True):
        assert self.random_policy_index is not None

        feed = dict(zip([self.private_ph, self.temp_ph, self.mask_ph]+self.last_act_ph_n_policy,[private,temp, mask]+last_action_n))
        if in_state is not None:
            feed[self.in_state_ph] = in_state

        a, out_state, ent, logits, a_cont = sess.run([self.policy[self.random_policy_index]['train']['act_sample'],
                             self.policy[self.random_policy_index]['train']['out_state'],
                             self.policy[self.random_policy_index]['train']['entropy'],
                                              self.policy[self.random_policy_index]['train']['logits'],
                                              self.policy[self.random_policy_index]['train']['act_sample_cont']] if train else
                            [self.policy[self.random_policy_index]['target']['act_sample'],
                             self.policy[self.random_policy_index]['target']['out_state'],
                            self.policy[self.random_policy_index]['target']['entropy'],
                             self.policy[self.random_policy_index]['target']['logits'],
                             self.policy[self.random_policy_index]['target']['act_sample_cont']], feed_dict=feed)
        return a, out_state, ent, logits, a_cont
        # s = np.random.choice([0, 1], p=[eps, 1 - eps])
        # if train and s == 0:
        #     ## random
        #     randa = np.random.choice(self.config.num_actions, size=a.shape)
        #
        #     return randa, out_state, ent
        # else:
        #
        #     return a, out_state, ent





    def experience(self, private, last_action, action, rewards, terminal):
        # Store transition in the replay buffer.
        assert self.random_policy_index is not None
        self.policy[self.random_policy_index]['buffer']['buffer'].add(private, last_action, action, rewards, terminal)

    def preupdate(self):
        self.policy[self.random_policy_index]['buffer']['replay_sample_index'] = None

    def update_pi(self, sess, private, last_action_n:list, action_n:list, mask, temp):
        # train p network
        #
        # q = self.get_q(sess, menu, workplace_embed, action, goal, train=False)

        feed_dict = dict(zip([self.private_ph, self.temp_ph, self.mask_ph]+self.last_act_ph_n_policy +self.act_ph_n_q,
                             [private, temp, mask]+last_action_n+action_n))
        _, p_loss = sess.run([self.policy[self.random_policy_index]['opt']['train_opt'],
                              self.policy[self.random_policy_index]['opt']['loss']
                               ], feed_dict=feed_dict)
        if np.isnan(p_loss):
            pass
        return p_loss

    def update_q(self, sess, agents, private, last_action_n:list, action_n:list, rewards, terminal, mask_all_agents:list, temp):
        # train q network

        ## td lambda

        bs, traj_len = private.shape[:2]

        ## get td lambda
        all_target_act = [agent.action(sess, private, last_action_n, temp, ~terminal, train=False)[0] for agent in agents]
        for act,m in zip(all_target_act, mask_all_agents):
            act[~m] = -1.
        qvalues = self.get_q(sess, private, last_action_n, all_target_act, train=False)
        r = rewards[:,-1:]
        td_target = [r]
        for i in reversed(range(traj_len-1)):
            q_ = qvalues[:,i+1:i+2]
            r = rewards[:,i:i+1] + self.config.gamma * (self.config.lamb * r + (1 - self.config.lamb) * q_) * (1. - terminal[:,i+1:i+2])

            td_target.append(r)

        td_target.reverse()
        td_target = np.concatenate(td_target, axis=1)

        # Break correlation
        if self.config.break_correlation:
            raise ValueError('Cannot break corr')
            # newbs = private.shape[0]*private.shape[1]
            # ind = np.random.choice(newbs, size=newbs//2, replace=False)
            # private_ = private.reshape((newbs,1,private.shape[2]))[ind] #menu
            # td_target_ = td_target.reshape((newbs,1))[ind]
            # terminal_ = terminal.reshape((newbs,1))[ind]
            # last_action_n_ = [action.reshape((newbs,1))[ind] for action in last_action_n]
            # action_n_ = [action.reshape((newbs,1))[ind] for action in action_n]
        else:
            private_ = private
            td_target_ = td_target
            terminal_ = terminal
            last_action_n_ = last_action_n
            action_n_ = action_n

        feed_dict = dict(zip([self.private_ph, self.target_q_ph, self.mask_ph]+self.last_act_ph_n_policy+self.act_ph_n_q,
                             [private_, td_target_, ~terminal_]+last_action_n_+action_n_))
        _, q_loss, out1, out2 = sess.run([self.c_train_opt, self.c_loss, self.train_q_vars, self.target_q_vars], feed_dict=feed_dict)

        if np.isnan(q_loss):
            pass

        return q_loss, td_target

    def update(self, sess, agents, temp, t):
        if not self.policy[self.random_policy_index]['buffer']['buffer'].is_available(): # replay buffer is not large enough
            return

        self.policy[self.random_policy_index]['buffer']['replay_sample_index'] = \
            self.policy[self.random_policy_index]['buffer']['buffer'].make_index(self.config.batch_size)
        # collect replay sample from all agents
        index = self.policy[self.random_policy_index]['buffer']['replay_sample_index']

        private_n, last_act_n, act_n, rewards_n, terminal_n \
            = self.policy[self.random_policy_index]['buffer']['buffer'].sample_index(index)

        private, rewards, terminal = \
            private_n[self.agent_index], rewards_n[self.agent_index], \
            terminal_n[self.agent_index]

        #mask
        mask_all_agents = []
        for i in range(self.n):
            m = np.arange(i%self.n, self.config.max_traj_len, self.n)
            tmp = np.zeros_like(terminal)
            tmp[:,m] = 1.
            mask = tmp * (1-terminal)
            mask_all_agents.append(mask.astype(np.bool, copy=False))
        my_mask = mask_all_agents[self.agent_index]

        q_loss, target_q  = \
            self.update_q(sess, agents, private, last_act_n, act_n, rewards, terminal, mask_all_agents, temp)

        # if t < 1e4:
        #     return [q_loss, None, np.mean(target_q[my_mask]), np.mean(rewards[my_mask]), np.std(target_q[my_mask]), np.sum(~terminal, axis=-1).mean()]

        # train p network
        p_loss = self.update_pi(sess, private, last_act_n, act_n, my_mask, temp)

        if t % self.config.update_step == 0:
            self.q_update(sess)
            self.p_update(sess)

        return [q_loss, p_loss, np.mean(target_q[my_mask]), np.mean(rewards[my_mask]), np.std(target_q[my_mask]), np.sum(~terminal, axis=-1).mean()]

    def p_update(self, sess):
        sess.run([self.policy[i]['update'] for i in range(self.config.num_ensemble)])

    def q_update(self, sess):
        sess.run(self.update_target_q)

    def get_q(self, sess, private, last_action_n:list, act_n:list, train=True):

        '''
        Q: [bs, time]
        '''

        q = sess.run(self.train_q if train else self.target_q,
                          feed_dict=dict(zip([self.private_ph]+self.last_act_ph_n_policy+self.act_ph_n_q,
                                                [private]+last_action_n+act_n)))
        return q

    def get_all_action_q(self, sess, private, last_action_n:list, train=True):
        q = []
        bs, t = private.shape[0:2]
        for i in range(self.config.num_actions):
            act = np.full((bs,t), i)
            act_n = [np.full(act.shape, -1) if k != self.agent_index else act for k in range(self.n)]
            q_ = self.get_q(sess, private,last_action_n, act_n, train)
            q.append(q_)

        return np.stack(q, axis=2)
    def get_pi_logit(self, sess, private, last_act_n:list, train=True):
        feed = dict(zip([self.private_ph]+self.last_act_ph_n_policy, [private]+last_act_n))
        logit = sess.run(self.policy[self.random_policy_index]['train']['logits'] if train else
                         self.policy[self.random_policy_index]['target']['logits'], feed_dict=feed)
        return np.squeeze(logit, axis=1)