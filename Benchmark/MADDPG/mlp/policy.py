
from Benchmark.MADDPG.mlp.buffer import ReplayBuffer
import tensorflow as tf
from tensorflow.contrib.distributions import RelaxedOneHotCategorical

def make_update_exp(vals, target_vals):
    polyak = 1.0 - 1e-2
    expression = []
    for var, var_target in zip(sorted(vals, key=lambda v: v.name), sorted(target_vals, key=lambda v: v.name)):
        expression.append(var_target.assign(polyak * var_target + (1.0-polyak) * var))
    expression = tf.group(*expression)
    return expression

class Policy:
    def __init__(self, name, num_agents, agent_index, K, p_func, q_func, p_input, temp_ph, act_dim, num_units, a_lr, buffer_size, train_q_scope='train_q'):
        self.name = name
        self.num_agents = num_agents
        self.agent_index = agent_index
        self.K = K
        self.p_func = p_func
        self.q_func = q_func
        self.a_lr = a_lr
        self.buffer_size = buffer_size
        self.train_q_scope = train_q_scope

        self.random = tf.placeholder(tf.int32, [])
        self.ensemble = self._build_net(p_input, temp_ph, act_dim, num_units)



    def __getitem__(self, i):
        return self.ensemble[i]

    def _build_net(self, p_input, temp_ph, act_dim, num_units):
        ensemble = []
        for i in range(self.K):
            logits, train_p_vars = self.p_func(p_input, act_dim, self.name+'_train_p_%d'%i, num_units, True)

            dist = RelaxedOneHotCategorical(temp_ph, logits=logits)

            act_sample_cont = dist.sample()
            act_sample = tf.argmax(act_sample_cont, axis=-1)

            ## target

            target_logits, target_p_vars = self.p_func(p_input, act_dim, self.name+'_target_p_%d'%i, num_units, False)

            update_target_p = make_update_exp(train_p_vars, target_p_vars)

            target_dist = RelaxedOneHotCategorical(temp_ph, logits=target_logits)
            target_act_sample_cont = target_dist.sample()
            target_act_sample = tf.argmax(target_act_sample_cont, axis=-1)

            ## loss

            act_input = [tf.zeros_like(act_sample_cont) for _ in range(self.num_agents)]
            act_input[self.agent_index] = act_sample_cont
            q_input = tf.concat([p_input] + act_input, 1)
            a_q, train_q_vars = self.q_func(q_input, 1, scope=self.train_q_scope, num_units=num_units, trainable=True)

            ## entropy
            prob = tf.nn.softmax(logits)
            entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=prob))
            ##target entropy
            target_prob = tf.nn.softmax(target_logits)
            target_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=target_logits, labels=target_prob))

            a_loss = -tf.reduce_mean(a_q) - 1e-2 * entropy
            a_train_opt = tf.train.AdamOptimizer(self.a_lr).minimize(a_loss, var_list=train_p_vars)


            ## replay buffer
            replay_buffer = ReplayBuffer(self.buffer_size, self.num_agents)
            replay_sample_index = None

            ensemble.append({'train':{'logits':logits, 'train_p_vars':train_p_vars, 'act_sample_cont':act_sample_cont,'act_sample':act_sample, 'dist':dist, 'entropy':entropy},
                             'target':{'logits':target_logits, 'train_p_vars':target_p_vars, 'act_sample_cont':target_act_sample_cont,'act_sample':target_act_sample,  'dist':target_dist, 'entropy':target_entropy},
                             'opt':{'loss':a_loss, 'train_opt':a_train_opt},
                             'buffer':{'buffer':replay_buffer, 'replay_sample_index':replay_sample_index},
                             'update':update_target_p,
                             'temp_ph':temp_ph})

        return ensemble

