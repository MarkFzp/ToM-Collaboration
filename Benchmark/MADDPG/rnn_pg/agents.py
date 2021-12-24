import numpy as np

from Benchmark.MADDPG.rnn_pg.policy import *

def p_func(inp, in_state, act_dim, scope, num_units, trainable=True, gpu=True):
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):

        if gpu:
            cell = tf.keras.layers.CuDNNGRU(num_units, return_sequences = True, return_state = True, trainable=trainable)
        else:
            cell = tf.keras.layers.GRU(num_units, return_sequences = True, return_state = True, trainable=trainable)

        out_seq, out_state = cell(inp, initial_state=in_state)

        L = tf.layers.dense(out_seq, num_units, activation=tf.nn.leaky_relu, trainable=trainable)
        L = tf.layers.dense(L, num_units, activation=tf.nn.leaky_relu, trainable=trainable)
        # L = tf.layers.dense(L, num_units, activation=tf.nn.leaky_relu, trainable=trainable)
        # L = tf.layers.dense(L, num_units, activation=tf.nn.leaky_relu, trainable=trainable)
        L = tf.layers.dense(L, act_dim, trainable=trainable)

    vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope)
    return [L, out_state], vars

def q_func(inp, q_dim, scope, num_units, trainable=True):
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        L = tf.layers.dense(inp, num_units, activation=tf.nn.leaky_relu, trainable=trainable)
        L = tf.layers.dense(L, num_units, activation=tf.nn.leaky_relu, trainable=trainable)
        # L = tf.layers.dense(L, num_units, activation=tf.nn.leaky_relu, trainable=trainable)
        # L = tf.layers.dense(L, num_units, activation=tf.nn.leaky_relu, trainable=trainable)
        # L = tf.layers.dense(L, num_units, activation=tf.nn.leaky_relu, trainable=trainable)
        L = tf.layers.dense(L, q_dim, trainable=trainable)

    vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope)
    return tf.squeeze(L, axis=-1), vars



def build_net(name, num_agents, index, num_dishes, num_ingredients, num_ensemble, act_dim, p_func, q_func, num_units,
              embed_dim, a_lr, c_lr, beta, buffer_size, max_traj_len, use_gpu):

    '''
    placeholder
    '''
    menu_ph = tf.placeholder(tf.float32, [None, None, num_dishes, num_ingredients])
    wpeb_ph = tf.placeholder(tf.float32, [None, None, num_ingredients] )

    goal_ph = tf.placeholder(tf.int32, [None])

    temp_ph = tf.placeholder(tf.float32, [])

    in_state_ph = tf.placeholder_with_default(tf.zeros([tf.shape(menu_ph)[0], num_units]),
                                              [None, num_units])

    mask_ph = tf.placeholder(tf.bool, [None, None]) ## ~terminal is not enough, alternating null actions are also important
    # if index == 0:
    #     # bind = tf.range(tf.shape(goal_ph)[0])
    #     # target_dish = tf.gather_nd(tf.transpose(menu_ph,[0,2,1,3]), tf.stack([bind, goal_ph], axis=1))
    #     obs_ph = tf.concat([tf.reshape(menu_ph-wpeb_ph[:,:,tf.newaxis,:], [-1, tf.shape(menu_ph)[1], (num_dishes)*num_ingredients]),
    #                             tf.tile(tf.one_hot(goal_ph, num_dishes)[:,tf.newaxis,:], [1,tf.shape(menu_ph)[1],1]),wpeb_ph
    #                             ], axis=-1)
    #     # obs_ph = target_dish-wpeb_ph #tf.layers.dense(target_dish-wpeb_ph, embed_dim, use_bias=False)
    #     # embed = tf.concat([target_dish[:,tf.newaxis,:], wpeb_ph[:,tf.newaxis,:]], axis=1)
    #     # obs_ph = tf.reshape(embed, [-1,2*embed_dim])
    #     # obs_ph = tf.reshape(embed, [-1, 2 * num_ingredients])
    #     # obs_ph = tf.concat([tf.reshape(embed, [-1, tf.shape(menu_ph)[1], (num_dishes)*embed_dim]),
    #     #                     tf.tile(tf.one_hot(goal_ph, num_dishes)[:,tf.newaxis,:], [1,tf.shape(menu_ph)[1],1]),
    #     #                     wpeb_ph], axis=-1)
    # else:
    #
    #     # embed = tf.layers.conv1d(tf.concat([menu_ph,wpeb_ph[:,tf.newaxis,:]], axis=1), embed_dim, 1,
    #     #                          use_bias=False)
    #     # embed  = tf.layers.dense(menu_ph-wpeb_ph[:,:,tf.newaxis,:], embed_dim, use_bias=False)
    #     obs_ph = tf.reshape(menu_ph-wpeb_ph[:,:,tf.newaxis,:], [-1, tf.shape(menu_ph)[1], (num_dishes)*num_ingredients])
    #     # obs_ph = tf.concat([tf.reshape(embed, [-1, tf.shape(embed)[1], (num_dishes) * embed_dim]),
    #     #                     wpeb_ph], axis=-1)
    # embed = tf.layers.dense(menu_ph-wpeb_ph[:,:,tf.newaxis,:], embed_dim, use_bias=False, activation=tf.nn.leaky_relu)
    # context_embed = tf.tile(tf.reduce_sum(embed, axis=-2, keepdims=True), (1, 1, tf.shape(menu_ph)[-2], 1))
    # embed = tf.concat([embed, context_embed], axis=-1)
    # embed = tf.layers.dense(embed, embed_dim, use_bias=False, activation=tf.nn.leaky_relu)
    embed = menu_ph - wpeb_ph[:, :, tf.newaxis, :]
    context_embed = tf.tile(tf.reduce_sum(embed, axis=-2, keepdims=True), (1, 1, tf.shape(menu_ph)[-2], 1))
    embed = tf.concat([embed, context_embed], axis=-1)
    if index == 0:
        bind = tf.range(tf.shape(goal_ph)[0])

        obs_ph = tf.gather_nd(tf.transpose(embed, [0, 2, 1, 3]), tf.stack([bind, goal_ph], axis=1))
        # obs_ph = tf.concat([tf.reshape(embed, [-1, tf.shape(menu_ph)[1], (num_dishes) * embed_dim]), tf.cast(tf.one_hot(goal_ph, num_dishes))], axis=-1)

        # obs_ph = target_dish-wpeb_ph
        # obs_ph = tf.reshape(embed, [-1,2*embed_dim])
        # obs_ph = tf.reshape(embed, [-1, 2 * num_ingredients])
        # obs_ph = tf.concat([tf.reshape(menu_ph-wpeb_ph[:,:,tf.newaxis,:], [-1, tf.shape(menu_ph)[1], (num_dishes)*num_ingredients]),
        #                     target_dish-wpeb_ph#tf.tile(tf.one_hot(goal_ph, num_dishes)[:,tf.newaxis,:], [1,tf.shape(menu_ph)[1],1]),wpeb_ph
        #                     ], axis=-1)
    else:

        # embed = tf.layers.conv1d(tf.concat([menu_ph,wpeb_ph[:,tf.newaxis,:]], axis=1), embed_dim, 1,
        #                          use_bias=False)

        obs_ph = tf.reduce_mean(context_embed, axis=2) #tf.reduce_sum(embed, axis=-2)

    act_ph_n = [tf.placeholder(tf.int32, [None, None]) for _ in range(num_agents)]

    target_q_ph = tf.placeholder(tf.float32, [None,None])

    rm_ph = tf.placeholder(tf.float32, [])
    rstd_ph = tf.placeholder(tf.float32, [])

    '''
    policy
    '''
    inp = {'p_input':obs_ph, 'in_state':in_state_ph, 'mask':mask_ph, 'temp':temp_ph, 'act_ph_n':act_ph_n, 'target_q_ph':target_q_ph,
           'rm_ph':rm_ph,'rstd_ph':rstd_ph}
    policy = Policy(name, num_agents, index, num_ensemble, p_func, q_func,inp, act_dim, num_units, a_lr, beta,
                    buffer_size, max_traj_len, name+'_train_q', name+'_target_q', use_gpu)

    '''
    qnet
    '''

    ## for critic
    q_input = tf.concat([obs_ph] + [tf.one_hot(act_ph, act_dim) for act_ph in act_ph_n], axis=-1)
    c_q, train_q_vars = q_func(q_input, 1, scope=name+"_train_q", num_units=num_units, trainable=True)

    c_loss = tf.reduce_mean(tf.square(tf.boolean_mask(c_q, mask_ph) - tf.boolean_mask(target_q_ph, mask_ph)))

    ## target
    target_q, target_q_vars = q_func(q_input, 1, scope=name+"_target_q", num_units=num_units, trainable=False)
    update_target_q = make_update_exp(train_q_vars, target_q_vars)
    '''
    critic train opt
    '''
    c_train_opt = tf.train.AdamOptimizer(c_lr).minimize(c_loss, var_list=train_q_vars)

    return {'ph': [menu_ph, wpeb_ph, goal_ph, act_ph_n, target_q_ph, temp_ph, in_state_ph, mask_ph, rm_ph, rstd_ph],
            'actor': policy,
            'critic': [c_q, target_q, c_loss, c_train_opt, update_target_q, train_q_vars, target_q_vars]}


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
            config.beta,
            config.buffer_size,
            config.max_traj_len,
            config.use_gpu
        )

        self.policy = dict['actor']
        self.train_q, self.target_q, self.c_loss, self.c_train_opt, self.update_target_q, self.train_q_vars, self.target_q_vars = dict['critic']

        self.menu_ph, self.wpeb_ph, self.goal_ph, self.act_ph_n, self.target_q_ph, self.temp_ph, \
        self.in_state_ph, self.mask_ph, self.rm_ph, self.rstd_ph = dict['ph']

        self.random_policy_index = None

    def set_policy_index(self):
        ## called before each episode
        self.random_policy_index = np.random.randint(self.config.num_ensemble)

    def action(self, sess, menu, wpeb, goal, temp, mask=np.array([[True]]), in_state=None,  train=True):
        assert self.random_policy_index is not None

        feed = {self.menu_ph: menu, self.wpeb_ph: wpeb, self.goal_ph: goal, self.temp_ph: temp, self.mask_ph: mask}
        if in_state is not None:
            feed[self.in_state_ph] = in_state

        a, out_state, ent, logits = sess.run([self.policy[self.random_policy_index]['train']['act_sample'],
                             self.policy[self.random_policy_index]['train']['out_state'],
                             self.policy[self.random_policy_index]['train']['entropy'],
                                              self.policy[self.random_policy_index]['train']['logits']] if train else
                            [self.policy[self.random_policy_index]['target']['act_sample'],
                             self.policy[self.random_policy_index]['target']['out_state'],
                            self.policy[self.random_policy_index]['target']['entropy'],
                             self.policy[self.random_policy_index]['target']['logits']], feed_dict=feed)
        return a, out_state, ent, logits



    def experience(self, menu, workplace_embed, action, rewards,  goal, terminal):
        # Store transition in the replay buffer.
        assert self.random_policy_index is not None
        self.policy[self.random_policy_index]['buffer']['buffer'].add(menu, workplace_embed, action, rewards,  goal, terminal)

    def preupdate(self):
        self.policy[self.random_policy_index]['buffer']['replay_sample_index'] = None

    def update_pi(self, sess, menu, workplace_embed, goal, action_n:list, rm, rstd, mask, temp):
        # train p network
        #
        # q = self.get_q(sess, menu, workplace_embed, action, goal, train=False)

        feed_dict = dict(zip([self.menu_ph, self.wpeb_ph, self.goal_ph, self.temp_ph, self.mask_ph, self.rm_ph, self.rstd_ph]+self.act_ph_n,
                             [menu, workplace_embed, goal, temp, mask, rm, rstd]+action_n))
        _, p_loss, out, out2, q = sess.run([self.policy[self.random_policy_index]['opt']['train_opt'],
                              self.policy[self.random_policy_index]['opt']['loss'],
                                   self.policy.log_prob, self.policy.log_prob_after_action, self.policy.q
                               ], feed_dict=feed_dict)
        if np.isnan(p_loss):
            pass
        return p_loss

    def update_q(self, sess, agents, menu, workplace_embed, action_n:list, rewards, goal, terminal, mask_all_agents:list):
        # train q network

        ## td lambda

        bs, traj_len = menu.shape[:2]

        ## get td lambda
        all_target_act = [agent.action(sess, menu, workplace_embed, goal, None, ~terminal, train=False)[0] for agent in
                          agents]
        for act, m in zip(all_target_act, mask_all_agents):
            act[~m] = -1.
        qvalues = [self.get_q(sess, menu[:, i:i + 1, :, :], workplace_embed[:, i:i + 1, :],
                              [act[:, i:i + 1] for act in all_target_act],
                              goal, train=False) for i in range(traj_len)]
        r = sr = rewards[:,-1:]
        discounted_r = [sr]
        td_target = [r]
        for i in reversed(range(traj_len-1)):
            q_ = qvalues[i+1]
            r = rewards[:,i:i+1] + self.config.gamma * (self.config.lamb * r + (1 - self.config.lamb) * q_) * (1. - terminal[:,i+1:i+2])

            td_target.append(r)

            sr = rewards[:,i:i+1] + self.config.gamma * sr * (1. - terminal[:,i+1:i+2])
            discounted_r.append(sr)

        td_target.reverse()
        td_target = np.concatenate(td_target, axis=1)

        discounted_r.reverse()
        discounted_r = np.concatenate(discounted_r, axis=1)

        ## normalize discounted rewards
        tdmean = np.mean(discounted_r[~terminal])#np.sum(discounted_r, axis=1, keepdims=True) / np.sum(~terminal, axis=1, keepdims=True)
        tdstd = np.std(discounted_r[~terminal])#np.tile(np.nanstd(np.where(np.isclose(discounted_r, 0), np.nan, discounted_r), axis=1, keepdims=True), (1,discounted_r.shape[1]))
        # new_tdtarget = td_target - tdmean
        # adv = np.divide(new_tdtarget, tdstd, out=np.zeros_like(new_tdtarget), where=tdstd!=0)
        # tdmean = np.mean(td_target[
        #                      ~terminal])  # np.sum(discounted_r, axis=1, keepdims=True) / np.sum(~terminal, axis=1, keepdims=True)
        # tdstd = np.std(td_target[~terminal])

        ## Break correlation
        if self.config.break_correlation:
            newbs = menu.shape[0]*menu.shape[1]
            ind = np.random.choice(menu.shape[0]*menu.shape[1], size=menu.shape[0]*menu.shape[1]//2, replace=False)
            menu_ = menu.reshape((newbs,1,menu.shape[2],menu.shape[3]))[ind] #menu
            workplace_embed_ = workplace_embed.reshape((newbs,1,workplace_embed.shape[2]))[ind]
            td_target_ = td_target.reshape((newbs,1))[ind]
            terminal_ = terminal.reshape((newbs,1))[ind]
            action_n_ = [action.reshape((newbs,1))[ind] for action in action_n]
            goal_ = np.tile(goal[:,np.newaxis], (1, menu.shape[1])).reshape((newbs,))[ind]
        else:
            menu_ = menu
            workplace_embed_ = workplace_embed
            td_target_ = td_target
            terminal_ = terminal
            action_n_ = action_n
            goal_ = goal

        feed_dict = dict(zip([self.menu_ph, self.wpeb_ph, self.goal_ph, self.target_q_ph, self.mask_ph]+self.act_ph_n,
                             [menu_, workplace_embed_, goal_, td_target_, ~terminal_]+action_n_))
        _, q_loss, out1, out2 = sess.run([self.c_train_opt, self.c_loss, self.train_q_vars, self.target_q_vars], feed_dict=feed_dict)

        if np.isnan(q_loss):
            pass

        return q_loss, td_target, tdmean, tdstd

    def update(self, sess, agents, temp, t):
        if not self.policy[self.random_policy_index]['buffer']['buffer'].is_available(): # replay buffer is not large enough
            return



        self.policy[self.random_policy_index]['buffer']['replay_sample_index'] = \
            self.policy[self.random_policy_index]['buffer']['buffer'].make_index(self.config.batch_size)
        # collect replay sample from all agents
        index = self.policy[self.random_policy_index]['buffer']['replay_sample_index']

        menu_n, wpeb_n, act_n, rewards_n, goal_n, terminal_n \
            = self.policy[self.random_policy_index]['buffer']['buffer'].sample_index(index)

        menu, wpeb, rewards, goal, terminal = \
            menu_n[self.agent_index], wpeb_n[self.agent_index], rewards_n[self.agent_index], \
            goal_n[self.agent_index], terminal_n[
                self.agent_index]

        # mask
        mask_all_agents = []
        for i in range(self.n):
            m = np.arange(i % self.n, self.config.max_traj_len, self.n)
            tmp = np.zeros_like(terminal)
            tmp[:, m] = 1.
            mask = tmp * (~terminal)
            mask_all_agents.append(mask)
        my_mask = mask_all_agents[self.agent_index]

        q_loss, target_q, rm, rstd  = \
            self.update_q(sess, agents, menu, wpeb, act_n, rewards, goal, terminal, mask_all_agents)

        # if t < 5e4:
        #     return [q_loss, 0, np.mean(target_q[mask]), np.mean(rewards[mask]), np.std(target_q[mask])]

        # train p network
        p_loss = self.update_pi(sess, menu, wpeb, goal, act_n, rm, rstd, my_mask, temp)

        if t % self.config.update_step == 0:
            # self.p_update(sess)
            self.q_update(sess)

        return [q_loss, p_loss, np.mean(target_q[my_mask]), np.mean(rewards[my_mask]), np.std(target_q[my_mask])]

    def p_update(self, sess):
        sess.run([self.policy[i]['update'] for i in range(self.config.num_ensemble)])

    def q_update(self, sess):
        sess.run(self.update_target_q)

    def get_q(self, sess, menu, wpeb, act_n:list, goal, train=True):

        '''
        Q: [bs, time]
        '''

        q = sess.run(self.train_q if train else self.target_q,
                          feed_dict=dict(zip([self.menu_ph, self.wpeb_ph, self.goal_ph]+self.act_ph_n,
                                                [menu, wpeb, goal]+act_n)))
        return q

    def get_all_action_q(self, sess, menu, wpeb, goal, train=True):
        q = []
        bs, t = menu.shape[0:2]
        for i in range(self.config.num_ingredients):
            act = np.full((bs,t), i)
            act_n = [np.full(act.shape, -1) if k != self.agent_index else act for k in range(self.n)]
            q_ = self.get_q(sess, menu, wpeb, act_n, goal, train)
            q.append(q_)

        return np.stack(q, axis=2)
    def get_pi_logit(self, sess, menu, wpeb, goal, train=True):
        logit = sess.run(self.policy[self.random_policy_index]['train']['logits'] if train else
                         self.policy[self.random_policy_index]['target']['logits'], feed_dict={self.menu_ph:menu,
                                                                          self.wpeb_ph: wpeb,
                                                                          self.goal_ph: goal})
        return np.squeeze(logit, axis=1)

    def clear_buffer(self):
        self.policy[self.random_policy_index]['buffer']['buffer'].clear()