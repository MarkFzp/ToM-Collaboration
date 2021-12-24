import tensorflow as tf
import numpy as np

import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
from value import ValueNetwork
from policy import PolicyNetwork
from belief import BeliefNetwork

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir))
from Game.calendar import Calendar
import pdb

class Agent:
    def __init__(self, sess, configs, name, have_private_self, have_private_other, is_train):
        self.sess_ = sess
        self.training_config_, self.game_config_,\
        self.value_config_, self.policy_config_,\
        self.belief_config_ = configs
        self.name_ = name
        self.calendar_ = Calendar(self.name_, 'X', -1, -1, -1, self.game_config_.num_slots) #only need num slots
        self.no_op_ = tf.no_op()
        self.index_ = 0 if self.name_ == 'A' else 1
        self.have_private_self_ = have_private_self
        self.have_private_other_ = have_private_other
        self.is_train_ = is_train
        self.global_step_ = tf.Variable(0, trainable = False)
        self.replay_buffer_ = []

        with tf.variable_scope('%s_background_encoding' % self.name_, reuse = tf.AUTO_REUSE):
            self.calendar_encoding_ = tf.get_variable('calendar_encoding', shape = [1, self.game_config_.num_slots, 1, 20],
                                                    initializer = tf.random_normal_initializer(mean = 0, stddev = 1e-2))
            self.all_calendars_ = tf.constant(self.calendar_.tensor_, dtype = tf.float32) # (2 ** num_slots) X num_slots
            self.all_actions_encoding_ = tf.constant(self.calendar_.action_tensor_, dtype = tf.float32) # Num_actions X (2 * num_slots)
            self.fixed_encoding_ = tf.identity(self.calendar_encoding_)
            self.fixed_feat_tensors_ = [self.fixed_encoding_] # 1 X num_slots X 1 X dim
            for idx, out_dim in enumerate(self.game_config_.fixed_fc_layer_info):
                fc = tf.layers.conv2d(self.fixed_feat_tensors_[-1], out_dim, kernel_size = 1, strides = 1, padding = 'VALID', 
                            activation = tf.nn.leaky_relu, kernel_initializer = tf.random_normal_initializer(mean = 0, stddev = 1e-2))
                self.fixed_feat_tensors_.append(fc)

            for j, cl in enumerate(self.game_config_.fixed_context_length):
                context = tf.tile(tf.reduce_sum(self.fixed_feat_tensors_[-1], axis = 1, keepdims = True), [1, self.game_config_.num_slots, 1, 1])
                context_fc = tf.layers.conv2d(context, cl, kernel_size = 1, strides = 1, padding='VALID', 
                            activation = tf.nn.leaky_relu, kernel_initializer = tf.random_normal_initializer(mean = 0, stddev = 1e-2))
                with_context = tf.concat([self.fixed_feat_tensors_[-1], context_fc], axis = 3)
                self.fixed_feat_tensors_.append(tf.layers.conv2d(with_context, self.game_config_.fixed_single_length[j],
                                                        kernel_size = 1, strides = 1, padding='VALID', 
                                                        activation = tf.nn.leaky_relu if j != len(self.game_config_.fixed_context_length) - 1 else None,
                                                        kernel_initializer = tf.random_normal_initializer(mean = 0, stddev = 1e-2)))
        self.background_encoder_varlist_ = [v for v in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)\
                                            if v.name.startswith('%s_background_encoding' % self.name_)]
        self.history_encoding_ = tf.placeholder(tf.float32, shape = [None, None, int(self.all_actions_encoding_.shape[1]) + 1], name = 'history_encoding')


        fixed_feat_tensor_trans = tf.transpose(self.fixed_feat_tensors_[-1], [0, 2, 1, 3])
        self.mean_background_ = tf.reduce_mean(fixed_feat_tensor_trans, 2) * tf.ones_like(self.history_encoding_)[:, :, 0: 1]
        self_unique_tensor = self.mean_background_
        other_unique_tensor = self.mean_background_
        if self.have_private_self_:# this belief should be binary multi-hot, Batch_Size X Sequence_len X private_dim
            self.private_belief_self_ = tf.placeholder(tf.float32, shape = [None, None, self.game_config_.num_slots],
                                                       name = 'private_belief_self')
            self_unique_tensor = tf.reduce_sum(tf.multiply(tf.expand_dims(self.private_belief_self_, -1), fixed_feat_tensor_trans), axis = 2)
        if self.have_private_other_:
            self.private_belief_other_ = tf.placeholder(tf.float32, shape = [None, None, self.game_config_.num_slots],
                                                        name = 'private_belief_other')
            other_unique_tensor = tf.reduce_sum(tf.multiply(tf.expand_dims(self.private_belief_other_, -1), fixed_feat_tensor_trans), axis = 2)
        self.agents_unique_tensors_  = [self_unique_tensor, other_unique_tensor] if self.index_ == 0 \
                                       else [other_unique_tensor, self_unique_tensor]

        #################
        ###   B-Net   ###
        #################
        self.other_agent_belief_tensor_ = None
        
        if self.have_private_self_:
            initial_belief = tf.constant(np.ones([1, 2 ** self.game_config_.num_slots]) / 2 ** self.game_config_.num_slots, dtype = tf.float32)
            self.initial_belief_ = initial_belief * tf.ones_like(self.history_encoding_)[:, 0: 1, 0: 1]
            with tf.variable_scope('%s_Belief' % self.name_, reuse = tf.AUTO_REUSE):
                self.belief_network_ = BeliefNetwork(self.belief_config_, self.calendar_.num_actions_,
                                                     self.all_calendars_, tf.concat(self.agents_unique_tensors_, axis = 2),
                                                     self.history_encoding_, self.initial_belief_)
            
            # Batch_Size X Sequence_len X num_action X dim
            other_belief_tensor = tf.reduce_sum(tf.multiply(tf.expand_dims(fixed_feat_tensor_trans, 2),
                                                            tf.expand_dims(self.belief_network_.new_belief_, -1)), axis = 3)
            self.other_agent_belief_tensor_ = tf.identity(other_belief_tensor)
            
            self.belief_spvs_ = tf.placeholder(tf.float32, shape = [None, self.game_config_.num_slots], name = 'belief_spvs')
            self.belief_spvs_mask_1_ = tf.placeholder(tf.float32, shape = [None, None, self.calendar_.num_actions_], name = 'belief_spvs_mask_1')
            self.belief_spvs_mask_2_ = tf.placeholder(tf.int32, shape = [None, 1], name = 'belief_spvs_mask_2')
            relevant_belief = tf.reduce_sum(self.belief_network_.new_belief_ * tf.expand_dims(self.belief_spvs_mask_1_, -1), axis = 2)
            self.belief_varlist_ = [v for v in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES) if v.name.startswith('%s_Belief' % self.name_)]
            self.belief_regularization_ = tf.add_n([ tf.nn.l2_loss(v) for v in self.belief_varlist_ if 'bias' not in v.name ])
            self.cross_entropy_func = lambda x, y: -1 * tf.reduce_mean(tf.reduce_sum(tf.multiply(x, tf.math.log(y + 1e-9)), axis = 1))
            self.output_beliefs_ = tf.gather_nd(tf.reshape(relevant_belief, [-1, self.game_config_.num_slots]), self.belief_spvs_mask_2_)
            #self.belief_loss_1_ = tf.nn.sigmoid_cross_entropy_with_logits(labels = self.belief_spvs_, logits = self.output_beliefs_)
            self.belief_loss_1_ = tf.reduce_sum(tf.square(self.belief_spvs_ - self.output_beliefs_), axis = 1)
            self.belief_loss_ = tf.reduce_mean(self.belief_loss_1_) + 1e-4 * self.belief_regularization_
            

        #################
        ###   Q-Net   ###
        #################
        self.state_actions_ = self.all_actions_encoding_ * tf.ones_like(tf.expand_dims(self.history_encoding_, -1))[:, :, 0: 1, :]
        with tf.variable_scope('%s_Q_primary' % self.name_, reuse = tf.AUTO_REUSE):
            self.q_primary_ = ValueNetwork(self.value_config_, self.state_actions_, self.history_encoding_,
                                           self.agents_unique_tensors_[0], self.agents_unique_tensors_[1],
                                           self.other_agent_belief_tensor_)
        with tf.variable_scope('%s_Q_target' % self.name_, reuse = tf.AUTO_REUSE):
            self.q_target_ = ValueNetwork(self.value_config_, self.state_actions_, self.history_encoding_,
                                           self.agents_unique_tensors_[0], self.agents_unique_tensors_[1],
                                           self.other_agent_belief_tensor_)

        self.q_values_spvs_ = tf.placeholder(tf.float32, shape = [None, None], name = 'q_values_spvs')
        self.q_values_spvs_mask_ = tf.placeholder(tf.float32, shape = [None, None, self.calendar_.num_actions_], name = 'q_value_spvs_mask')
        self.q_primary_varlist_ = [v for v in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)\
                                     if v.name.startswith('%s_Q_primary' % self.name_)]
        self.q_target_varlist_ = [v for v in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)\
                                     if v.name.startswith('%s_Q_target' % self.name_)]
        self.q_net_regularization_ = tf.add_n([tf.nn.l2_loss(v) for v in self.q_primary_varlist_ if 'bias' not in v.name])
        self.q_learning_loss_ = tf.div_no_nan(tf.reduce_sum(tf.square(tf.reduce_sum(self.q_primary_.values_ * self.q_values_spvs_mask_, axis = 2)\
                                                        - self.q_values_spvs_)),  tf.reduce_sum(self.q_values_spvs_mask_))\
                                                        + 1e-4 * self.q_net_regularization_

        self.q_target_update_ = tf.group([v_t.assign(v_t * (1 - 0.2) + v * 0.2)\
                                          for v_t, v in zip(self.q_target_varlist_, self.q_primary_varlist_)])
        #################
        ###   P-Net   ###
        #################
        with tf.variable_scope('%s_Policy' % self.name_, reuse = tf.AUTO_REUSE):
            self.policy_network_ = PolicyNetwork(self.policy_config_, self.calendar_.num_actions_, other_unique_tensor, self.history_encoding_)
        self.policy_spvs_ = tf.placeholder(tf.int32, shape = [None, self.calendar_.num_actions_], name = 'policy_spvs')
        self.policy_spvs_mask_ = tf.placeholder(tf.int32, shape = [None, 1], name = 'policy_spvs_mask')
        self.policy_varlist_ = [v for v in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES) if v.name.startswith('%s_Policy' % self.name_)]
        self.policy_regularization_ = tf.add_n([ tf.nn.l2_loss(v) for v in self.policy_varlist_ if 'bias' not in v.name ])
        self.policy_loss_ = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(tf.reshape(self.policy_spvs_, [-1, self.calendar_.num_actions_]),
                                                                                      tf.gather_nd(tf.reshape(self.policy_network_.action_prob_unscale_,
                                                                                                   [-1, self.calendar_.num_actions_]), self.policy_spvs_mask_))) +\
                            1e-4 * self.policy_regularization_

        self.total_loader_ = tf.train.Saver([v for v in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)\
                                            if 'Adam' not in v.name and v.name.startswith(self.name_)])
        self.total_saver_ = tf.train.Saver([v for v in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)\
                                            if v.name.startswith(self.name_)], max_to_keep = None)

        if self.is_train_:
            self.opt_q_ = tf.train.AdamOptimizer(learning_rate = self.training_config_.q_learning_rate)
            self.q_training_op_ = self.opt_q_.minimize(self.q_learning_loss_, global_step = self.global_step_,
                                                       var_list = self.q_primary_varlist_ + self.background_encoder_varlist_)
            self.opt_policy_ = tf.train.AdamOptimizer(learning_rate = self.training_config_.policy_learning_rate)
            self.policy_training_op_ = self.opt_policy_.minimize(self.policy_loss_, global_step = self.global_step_,
                                                                 var_list = self.policy_varlist_ + self.background_encoder_varlist_)
            if self.have_private_self_:
                self.opt_belief_ = tf.train.AdamOptimizer(learning_rate = self.training_config_.belief_learning_rate)
                self.belief_training_op_ = self.opt_policy_.minimize(self.belief_loss_, global_step = self.global_step_,
                                                                 var_list = self.belief_varlist_ + self.background_encoder_varlist_)
    
    def collect(self, trajectory):
        if len(self.replay_buffer_) == self.training_config_.memory_size:
            i = 0
            while True:
                if i == len(self.replay_buffer_):
                    i = 0
                if self.replay_buffer_[i][-1][1] == self.game_config_.success_reward:
                    self.replay_buffer_.pop(i)
                    break
                else:
                    if self.training_config_.get('bad_pop_prob'):
                        pop_prob = self.training_config_.bad_pop_prob
                    else:
                        pop_prob = 1.0
                    rd = np.random.choice(2, 1, p = [1 - pop_prob, pop_prob])
                    if rd == 1:
                        self.replay_buffer_.pop(i)
                        break
                    else:
                        i += 1

        self.replay_buffer_.append(trajectory)

    def target_net_update(self):
        self.sess_.run(self.q_target_update_)
    #     if soft:
    #         self.sess_.run([v_t.assign(v_t * (1 - 0.2) + v * 0.2) for v_t, v in\
    #                         zip(self.q_target_varlist_, self.q_primary_varlist_)])
    #     else:
    #         self.sess_.run([v_t.assign(v) for v_t, v in\
    #                         zip(self.q_target_varlist_, self.q_primary_varlist_)])

    def save_ckpt(self, ckpt_dir, global_step):
        if not os.path.exists(ckpt_dir):
            os.makedirs(ckpt_dir)
        if not os.path.exists(os.path.join(ckpt_dir, self.name_)):
            os.makedirs(os.path.join(ckpt_dir, self.name_))

        self.total_saver_.save(self.sess_, os.path.join(ckpt_dir, self.name_, 'checkpoint'), global_step = global_step)
        print('Saved agent %s <%d> ckpt to %s' % (self.name_, global_step, ckpt_dir))
    
    def restore_ckpt(self, ckpt_dir):
        ckpt_status = tf.train.get_checkpoint_state(os.path.join(ckpt_dir, self.name_))
        if ckpt_status:
            self.total_loader_.restore(self.sess_, ckpt_status.model_checkpoint_path)
        if ckpt_status:
            print('%s Load model from %s' % (self.name_, ckpt_status.model_checkpoint_path))
            return True
        print('%s Fail to load model from %s' % (self.name_, ckpt_dir))
        return False

    def look(self, previous_state, last_action_index):
        belief_history_state_encoding = np.tile(previous_state['belief_history_state_encoding'],
                                                [self.calendar_.num_calendars_, 1])
        value_history_state_encoding = np.tile(previous_state['value_history_state_encoding'],
                                                [self.calendar_.num_calendars_, 1])
        policy_history_state_encoding = np.tile(previous_state['policy_history_state_encoding'],
                                                [self.calendar_.num_calendars_, 1])
        history_encoding = np.tile(previous_state['last_action'], [self.calendar_.num_calendars_, 1, 1])
        target_encoding = np.expand_dims(self.calendar_.tensor_, 1)
        #self.private_dim X 1 X num_actions
        value_history_state, action_prob, action_prob_unscale, policy_history_state, belief_history_state =\
            self.sess_.run([self.q_primary_.history_states_, self.policy_network_.action_prob_,
                            self.policy_network_.action_prob_unscale_, self.policy_network_.history_states_,
                            self.belief_network_.history_states_], 
                            {self.q_primary_.history_state_input_: value_history_state_encoding,
                            self.policy_network_.history_state_input_: policy_history_state_encoding,
                            self.belief_network_.history_state_input_: belief_history_state_encoding,
                            self.history_encoding_: history_encoding,
                            self.private_belief_other_: target_encoding})
        likelihood = action_prob[:, 0, last_action_index] + 1e-9
        return value_history_state[0: 1, ...], policy_history_state[0: 1, ...], belief_history_state[0: 1, ...], likelihood

    # this function is used to do inference and play the game, not for training
    def play(self, previous_state, epsilon):
        value_history_state_encoding = np.tile(previous_state['value_history_state_encoding'],
                                                [self.calendar_.num_calendars_, 1])
        policy_history_state_encoding = np.tile(previous_state['policy_history_state_encoding'],
                                                [self.calendar_.num_calendars_, 1])
        belief_history_state_encoding = np.tile(previous_state['belief_history_state_encoding'],
                                                [self.calendar_.num_calendars_, 1])
        history_encoding = np.tile(previous_state['last_action'], [self.calendar_.num_calendars_, 1, 1])
        belief_other = previous_state['target_belief'] # self.prviate_dim
        self_target_encoding = np.tile(previous_state['target_encoding'], [self.calendar_.num_calendars_, 1, 1])
        other_target_encoding = np.expand_dims(self.calendar_.tensor_, 1)
        #self.private_dim X 1 X num_actions
        action_qs_raw, value_history_state, policy_history_state, belief_history_state, belief_estimate =\
            self.sess_.run([self.q_primary_.values_, self.q_primary_.history_states_,
                            self.policy_network_.history_states_, self.belief_network_.history_states_,
                            self.belief_network_.new_belief_],
                            {self.q_primary_.history_state_input_: value_history_state_encoding,
                            self.policy_network_.history_state_input_: policy_history_state_encoding,
                            self.belief_network_.history_state_input_: belief_history_state_encoding,
                            self.history_encoding_: history_encoding,
                            self.private_belief_self_: self_target_encoding,
                            self.private_belief_other_: other_target_encoding})
        value_history_state = value_history_state[0: 1, ...]
        policy_history_state = policy_history_state[0: 1, ...]
        belief_history_state = belief_history_state[0: 1, ...]
        action_qs = np.sum(np.expand_dims(belief_other, 1) * np.squeeze(action_qs_raw), axis = 0)
        rd = np.random.choice(2, 1, p = [1 - epsilon, epsilon])
        if rd == 1 and self.is_train_:
            action_idx = np.random.randint(self.calendar_.num_actions_)
        elif self.is_train_:
            if self.game_config_.get('acting_boltzman_beta'):
                beta = self.game_config_.acting_boltzman_beta
            else:
                beta = self.game_config_.acting_boltzman_beta_learn if epsilon != 0 else self.game_config_.acting_boltzman_beta_help
            expq = np.exp(beta * (np.squeeze(action_qs) - np.max(action_qs)))
            act_prob = expq / np.sum(expq)
            action_idx = int(np.random.choice(self.calendar_.num_actions_, 1, p = act_prob))
        else:
            maximums = np.argwhere(action_qs == np.amax(action_qs)).flatten().tolist()
            action_idx = int(np.random.choice(maximums, 1))
        return action_idx, value_history_state, policy_history_state, belief_history_state, action_qs_raw, belief_estimate

def main():
    from easydict import EasyDict as edict
    calendar = Calendar('A', 'B', 0, -1, 1, 10)
    configs = [edict({'batch_size': 128,
                      'q_learning_rate': 1e-3,
                      'policy_learning_rate': 1e-3,
                      'belief_learning_rate': 1e-3,
                      'memory_size': 1e3}),
               edict({'num_slots': 10,
                      'action_coding_length': 10,
                      'acting_boltzman_beta': 10,
                      'fixed_fc_layer_info': [20, 10],
                      'fixed_single_length': [20, 10],
                      'fixed_context_length': [8, 6]}),
               edict({'history_dim': 10, 'fc_layer_info': [20, 15, 10]}),
               edict({'history_dim': 10, 'fc_layer_info': [20, 15, 10]}),
               edict({'history_dim': 10, 'fc_layer_info': [20, 15, 10]}),
               edict({'history_dim': 20, 'fc_layer_info': [30, 20]})]
    import copy
    sess = tf.Session()
    agentA = Agent(sess, configs, 'A', True, True, True)
    agentB = Agent(sess, configs, 'B', True, False, False)
    init = tf.global_variables_initializer()
    sess.run(init)

    # example_game_step = {'history_state_encoding': np.zeros([1, agentA_infer.value_config_.history_dim]),
    #                      'last_action': np.zeros([1, 1, agentA_infer.game_config_.num_actions]),
    #                      'menu': np.random.randint(0, 10, size = [1, agentA_infer.game_config_.num_dishes,
    #                                                               agentA_infer.game_config_.private_coding_length]),
    #                      'target_encoding': np.array([[[0, 0, 1, 0]]]),
    #                      'observation_encoding': np.zeros([1, 1, agentA_infer.game_config_.state_coding_length])}
    # print(agentA_infer.play(example_game_step, 0))
    # example_game_step_B = {'history_state_encoding': np.zeros([1, agentA_infer.value_config_.history_dim]),
    #                        'last_action': np.zeros([1, 1, agentA_infer.game_config_.num_actions]),
    #                        'menu': np.random.randint(0, 10, size = [1, agentA_infer.game_config_.num_dishes,
    #                                                                 agentA_infer.game_config_.private_coding_length]),
    #                        'target_belief': np.array([0.25, 0.25, 0.25, 0.25]),
    #                        'observation_encoding': np.zeros([1, 1, agentA_infer.game_config_.state_coding_length])}
    # print(agentB_infer.play(example_game_step_B, 0))

if __name__ == '__main__':
    main()
