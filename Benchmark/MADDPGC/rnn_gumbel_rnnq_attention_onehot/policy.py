
from Benchmark.MADDPGC.rnn_gumbel_rnnq_attention_onehot.buffer import ReplayBuffer
import tensorflow as tf
from tensorflow.contrib.distributions import Categorical

def make_update_exp(vals, target_vals):
    polyak = 0 #1.0 - 1e-2
    expression = []
    for var, var_target in zip(sorted(vals, key=lambda v: v.name), sorted(target_vals, key=lambda v: v.name)):
        expression.append(var_target.assign(polyak * var_target + (1.0-polyak) * var))
    expression = tf.group(*expression)
    return expression

class Policy:
    def __init__(self, name, num_agents, agent_index, K, p_func, q_func, inp, act_dim, num_units, a_lr, beta, buffer_size, max_traj_len, train_q_scope='train_q', target_q_scope='target_q', use_gpu=True):
        self.name = name
        self.num_agents = num_agents
        self.agent_index = agent_index
        self.K = K
        self.p_func = p_func
        self.q_func = q_func
        self.a_lr = a_lr
        self.buffer_size = buffer_size
        self.max_traj_len = max_traj_len
        self.train_q_scope = train_q_scope
        self.target_q_scope = target_q_scope
        self.beta = beta
        self.use_gpu = use_gpu

        self.random = tf.placeholder(tf.int32, [])
        self.ensemble = self._build_net(inp, act_dim, num_units)



    def __getitem__(self, i):
        return self.ensemble[i]

    def _build_net(self, inp, act_dim, num_units):
        p_input = inp['p_input']
        in_state_ph = inp['in_state']
        bool_mask = inp['mask']
        act_ph_n_q = inp['act_ph_n_q']
        temp_ph = inp['temp']
        action_tensor = inp['action_tensor']
        calendar_tensor = inp['calendar_tensor']
        q_in_state_ph = inp['q_in_state_ph']


        ensemble = []
        for i in range(self.K):

            [logits, out_state], train_p_vars = self.p_func(p_input, in_state_ph, calendar_tensor, act_dim, self.name+'_train_p_%d'%i, num_units, True, self.use_gpu)
            prob = tf.nn.softmax(logits)

            # logits_flat = tf.reshape(logits, [tf.shape(logits)[0] * tf.shape(logits)[1], tf.shape(logits)[2]])
            # act_sample = tf.random.categorical(logits_flat, 1)
            # act_sample_dec = tf.reshape(act_sample, [tf.shape(logits)[0], tf.shape(logits)[1]])


            dist = tf.contrib.distributions.RelaxedOneHotCategorical(temp_ph, logits=logits)

            act_sample_cont = dist.sample()
            act_sample = tf.argmax(act_sample_cont, axis=-1)
            # act_sample_cont = tf.reduce_sum(
            #     act_sample_cont[:, :, :, tf.newaxis] *
            #     tf.tile(action_tensor[tf.newaxis, tf.newaxis,:,:], [tf.shape(act_sample_cont)[0], tf.shape(act_sample_cont)[1], 1, 1]),
            #     axis=-2)

            ## loss

            q_input = tf.concat([p_input]+[tf.one_hot(act, act_dim) if i!=self.agent_index else
                                           act_sample_cont
                                           for i, act in enumerate(act_ph_n_q)], axis=-1)
            [q, _], _ = self.q_func(q_input, q_in_state_ph, calendar_tensor, 1, scope=self.train_q_scope, num_units=num_units, trainable=True, gpu= self.use_gpu)

            m_pg = - tf.reduce_mean(tf.boolean_mask(q, bool_mask))
            reg = tf.reduce_mean(tf.square(logits))
            ## entropy
            entropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels=prob, logits=logits)
            entropy = tf.reduce_mean(tf.boolean_mask(entropy, bool_mask))

            a_loss = m_pg + self.beta * reg
            a_train_opt = tf.train.AdamOptimizer(self.a_lr).minimize(a_loss, var_list=train_p_vars)


            ## replay buffer
            replay_buffer = ReplayBuffer(self.buffer_size,self.num_agents, self.max_traj_len)
            replay_sample_index = None

            ## target
            [target_logits, target_out_state], target_p_vars = self.p_func(p_input, in_state_ph, calendar_tensor, act_dim,
                                                            self.name + '_target_p_%d' % i, num_units, False,
                                                            self.use_gpu)
            target_prob = tf.nn.softmax(target_logits)
            target_dist = tf.contrib.distributions.RelaxedOneHotCategorical(temp_ph, logits=target_logits)


            # target_logits_flat = tf.reshape(target_logits, [tf.shape(target_logits)[0] * tf.shape(target_logits)[1], tf.shape(target_logits)[2]])
            # target_act_sample = tf.random.categorical(target_logits_flat, 1)
            # target_act_sample_dec =  tf.reshape(target_act_sample, [tf.shape(target_logits)[0], tf.shape(target_logits)[1]])

            target_act_sample_cont = target_dist.sample()
            target_act_sample = tf.argmax(target_act_sample_cont, axis=-1)

            target_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels=target_prob, logits=target_logits)
            target_entropy = tf.reduce_mean(tf.boolean_mask(target_entropy, bool_mask))

            update_target_p = make_update_exp(train_p_vars, target_p_vars)

            ensemble.append({'train':{'logits':logits, 'train_p_vars':train_p_vars, 'act_sample_cont':act_sample_cont,
                                      'act_sample':act_sample, 'out_state':out_state, 'dist':dist, 'entropy':entropy,
                                      'prob':prob},
                             'target': {'logits': target_logits, 'train_p_vars': target_p_vars,
                                        'act_sample_cont': target_act_sample_cont,
                                        'act_sample': target_act_sample, 'out_state': target_out_state, 'dist': target_dist,
                                        'entropy': target_entropy,
                                        'prob': target_prob},
                             'opt':{'loss':a_loss, 'train_opt':a_train_opt},
                             'buffer':{'buffer':replay_buffer, 'replay_sample_index':replay_sample_index},
                            'update':update_target_p})

        return ensemble

