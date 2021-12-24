import copy
import numpy as np
import tensorflow as tf

from AgentC.agent import Agent
from Game.calendar import Calendar

import pdb

def play(agentA, agentB, game, epsilons, is_train = True):
    game.reset(train = is_train)
    agents = {agentA.name_: agentA, agentB.name_: agentB}
    initial_state = game.start()
    next_state = copy.deepcopy(initial_state)

    rd = np.random.choice(2, 1)
    actor = agentA.name_ if rd == 0 else agentB.name_
    observer = agentB.name_ if rd == 0 else agentA.name_
    agent_last_value_states = {agentA.name_: np.zeros([1, agentA.value_config_.history_dim]),
                               agentB.name_: np.zeros([1, agentB.value_config_.history_dim])}
    agent_last_belief_states = {agentA.name_: np.zeros([1, agentA.belief_config_.history_dim]),
                                agentB.name_: np.zeros([1, agentB.belief_config_.history_dim])}
    agent_last_policy_states = {agentA.name_: np.zeros([1, agentA.policy_config_.history_dim]),
                                agentB.name_: np.zeros([1, agentB.policy_config_.history_dim])}
    initial_belief = np.ones(game.num_calendars_)
    initial_belief /= game.num_calendars_
    agent_beliefs = {agentA.name_: copy.deepcopy(initial_belief), agentB.name_: copy.deepcopy(initial_belief)}
    
    last_action = -1
    last_belief_symbol = -1
    last_action_embedding = np.zeros([1, 1, game.num_slots_ * 2 + 1])
    last_action_embedding[0, 0, -1] = -1
    sar_embedding_sequence = [(np.expand_dims(np.expand_dims(next_state[agentA.name_].private, 0), 0),
                               np.expand_dims(np.expand_dims(next_state[agentB.name_].private, 0), 0),
                               last_action_embedding, game.meetable_slots_, rd)]
   
    no_step = 0
    while not next_state[actor].terminate:
        game_state_cache = {'value_history_state_encoding': agent_last_value_states[actor],
                            'policy_history_state_encoding': agent_last_policy_states[actor],
                            'belief_history_state_encoding': agent_last_belief_states[actor],
                            'last_action': last_action_embedding,
                            'target_belief': agent_beliefs[actor],
                            'target_encoding': np.expand_dims(np.expand_dims(initial_state[actor]['private'], 0), 0)}
        last_action, agent_last_value_states[actor], agent_last_policy_states[actor],\
        agent_last_belief_states[actor], action_qs, belief_estimate\
                = agents[actor].play(game_state_cache, epsilons[actor == agentB.name_])
        agent_last_value_states[observer], agent_last_policy_states[observer], agent_last_belief_states[observer], likelihood =\
            agents[observer].look({'value_history_state_encoding': agent_last_value_states[observer],
                                    'policy_history_state_encoding': agent_last_policy_states[observer],
                                    'belief_history_state_encoding': agent_last_belief_states[observer],
                                    'last_action': last_action_embedding},
                                    last_action)
        agent_beliefs[observer] *= likelihood
        agent_beliefs[observer] /= np.sum(agent_beliefs[observer])
        slot_belief = np.sum(game.tensor_ * np.expand_dims(agent_beliefs[observer], -1), 0)
        last_action_embedding = np.zeros([1, game.num_slots_ * 2 + 1])
        last_action_embedding[0, -1] = int(actor == agentB.name_)
        last_action_embedding[0, 0: -1] = game.action_tensor_[last_action, :]
        last_action_embedding = np.expand_dims(last_action_embedding, 0)
        oh_last_action_index = np.zeros([1, 1, game.num_actions_])
        np.put(oh_last_action_index, last_action, 1)
        #last_belief_embedding = np.expand_dims(np.expand_dims(np.array([np.random.choice(2, 1, p = [max(0, 1 - slot_belief[i]), slot_belief[i]])[0]\
        #                                                                for i in range(slot_belief.shape[0])]), 0), 0)
        last_belief_embedding = np.expand_dims(np.expand_dims(np.around(slot_belief, decimals = 1), 0), 0)
        #last_belief_embedding = np.expand_dims(np.expand_dims(slot_belief, 0), 0)
        next_state = game.proceed({actor: last_action})
        if is_train:
            sar_embedding_sequence.append(((last_action_embedding, oh_last_action_index), next_state[actor].reward, last_belief_embedding))
        else:
            sar_embedding_sequence.append(((last_action_embedding, oh_last_action_index), next_state[actor].reward,
                                        copy.deepcopy(agent_beliefs[observer]), action_qs, likelihood, belief_estimate, agent_beliefs[actor]))
        actor = agentA.name_ if actor == agentB.name_ else agentB.name_
        observer = agentA.name_ if observer == agentB.name_ else agentB.name_
        no_step += 1
        if no_step == game.num_slots_:
            break
    #[(calendarA, calendarB, meetable_slots, Null-action, first_actor), ([a, ai], r, m), ([a, ai], r, m), ...]
    return sar_embedding_sequence

def train_step(agent, global_step, switch = 5e4):
    samples_idx = np.random.choice(len(agent.replay_buffer_), agent.training_config_.batch_size, replace = False)
    samples = [agent.replay_buffer_[i] for i in samples_idx]
    samples_length = [len(agent.replay_buffer_[i]) for i in samples_idx]
    max_length = np.max(samples_length)
    previous_actions = np.zeros([agent.training_config_.batch_size, max_length,
                                 agent.game_config_.num_slots * 2 + 1])
    policy_rnn_states = np.zeros([agent.training_config_.batch_size, agent.policy_config_.history_dim])
    value_rnn_states = np.zeros([agent.training_config_.batch_size, agent.value_config_.history_dim])
    belief_rnn_states = np.zeros([agent.training_config_.batch_size, agent.belief_config_.history_dim])
    goal_embedding_self = np.zeros([agent.training_config_.batch_size, max_length, agent.game_config_.num_slots])
    goal_embedding_other = np.zeros([agent.training_config_.batch_size, max_length, agent.game_config_.num_slots])
    for i in range(len(samples)):
        for si in range(len(samples[i])):
            goal_embedding_self[i, si, :] = samples[i][0][agent.index_]
            goal_embedding_other[i, si, :] = samples[i][0][1 - agent.index_]

    action_masks = np.zeros([agent.training_config_.batch_size, max_length])
    action_spvs = []
    
    for i in range(len(samples)):
        for si in range(1, len(samples[i])):
            previous_actions[i, si, :] = samples[i][si][0][0]
        for si in range(len(samples[i]) - 1):
            if si % 2 != (samples[i][0][4] + agent.index_) % 2:
                action_masks[i, si] = 1
                action_spvs.append(samples[i][si + 1][0][1])
    action_masks = np.expand_dims(np.nonzero(np.reshape(action_masks,
                                                        [agent.training_config_.batch_size * max_length, 1]))[0], 1)
    if len(action_spvs) != 0:
        action_spvs = np.squeeze(np.array(action_spvs), axis = (1, 2))
        policy_training_op = agent.policy_training_op_
        policy_loss_op = agent.policy_loss_
        other_policy_est_op = agent.policy_network_.action_prob_
    else:
        action_spvs = np.zeros([0, agent.calendar_.num_actions_])
        policy_training_op = agent.no_op_
        policy_loss_op = agent.no_op_
        other_policy_est_op = agent.no_op_
    q_spvs = np.zeros([agent.training_config_.batch_size, max_length])
    q_spvs_mask = np.zeros([agent.training_config_.batch_size, max_length, agent.calendar_.num_actions_])
    belief_mask = np.zeros([agent.training_config_.batch_size, max_length])
    belief_spvs = []
    
    feed_dict = {agent.policy_network_.history_state_input_: policy_rnn_states,
                 agent.q_target_.history_state_input_: value_rnn_states,
                 agent.belief_network_.history_state_input_: belief_rnn_states,
                 agent.history_encoding_: previous_actions,
                 agent.policy_spvs_mask_: action_masks,
                 agent.policy_spvs_: action_spvs,
                 agent.private_belief_self_: goal_embedding_self,
                 agent.private_belief_other_: goal_embedding_other}
    
    _, policy_loss, action_target_qs, other_policy_est =\
        agent.sess_.run([policy_training_op, policy_loss_op,
                         agent.q_target_.values_, other_policy_est_op], feed_dict) 
    
    max_target_qs = np.max(action_target_qs, axis = 2)
    for i in range(len(samples)):
        for si, step in enumerate(samples[i][0: -1]):
            if si % 2 == (samples[i][0][4] + agent.index_) % 2:
                next_action_index = np.nonzero(samples[i][si + 1][0][1])[2][0]
                q_spvs_mask[i, si, next_action_index] = 1
                reward1 = samples[i][si + 1][1]
                reward2 = samples[i][si + 2][1] if si < len(samples[i]) - 2 else 0
                next_q = max_target_qs[i, si + 2] if si < len(samples[i]) - 3 else 0
                q_spvs[i, si] = reward1 + agent.training_config_.q_learning_gamma *\
                                (reward2 + agent.training_config_.q_learning_gamma * next_q)
                belief_mask[i, si] = 1
                belief_spvs.append(samples[i][si + 1][2])

    belief_mask = np.expand_dims(np.nonzero(np.reshape(belief_mask,
                                                        [agent.training_config_.batch_size * max_length, 1]))[0], 1)
    del feed_dict[agent.policy_network_.history_state_input_]
    del feed_dict[agent.q_target_.history_state_input_]
    del feed_dict[agent.policy_spvs_]
    del feed_dict[agent.policy_spvs_mask_]
    feed_dict[agent.q_primary_.history_state_input_] = value_rnn_states
    feed_dict[agent.q_values_spvs_mask_] = q_spvs_mask
    feed_dict[agent.q_values_spvs_] = q_spvs
    belief_train_op = agent.no_op_
    belief_loss_op = agent.no_op_
    likelihood_op = agent.no_op_
    slot_belief_op = agent.no_op_
    if global_step > 2 * switch and len(belief_spvs) > 0:
        belief_train_op = agent.belief_training_op_
        belief_loss_op = agent.belief_loss_
        likelihood_op = agent.belief_network_.likelihood_pre_
        slot_belief_op = agent.belief_network_.new_belief_
        feed_dict[agent.belief_spvs_mask_1_] = q_spvs_mask
        feed_dict[agent.belief_spvs_mask_2_] = belief_mask
        feed_dict[agent.belief_spvs_] = np.squeeze(np.array(belief_spvs), axis = (1, 2))

    _, _, q_predict, q_learning_loss, belief_predict_loss, likelihood, slot_belief =\
        agent.sess_.run([agent.q_training_op_, belief_train_op,
                         agent.q_primary_.values_, agent.q_learning_loss_,
                         belief_loss_op, likelihood_op, slot_belief_op], feed_dict)
    # if np.sum(np.isnan(agent.sess_.run(agent.q_primary_varlist_[-1]))) > 0:
    #     pdb.set_trace()
    
    if global_step % 1000 == 0:
        #if slot_belief is not None:
        #    pdb.set_trace()
        valid_msg_q_spvs = []
        invalid_msg_q_spvs = []
        valid_prop_q_spvs = []
        invalid_prop_q_spvs = []
        related_indices = np.nonzero(q_spvs_mask)
        for i in range(related_indices[0].shape[0]):
            j = related_indices[1][i]
            if related_indices[2][i] < agent.calendar_.num_msgs_:
                valid = (np.sum((samples[related_indices[0][i]][0][agent.index_][0, 0, ...] -\
                        agent.calendar_.action_tensor_[related_indices[2][i], 0: agent.calendar_.num_slots_]) == -1) == 0)
                if valid:
                    valid_msg_q_spvs.append(q_spvs[related_indices[0][i], j])
                else:
                    invalid_msg_q_spvs.append(q_spvs[related_indices[0][i], j])
            else:
                valid = (np.sum((1 - agent.calendar_.action_tensor_[related_indices[2][i], agent.calendar_.num_slots_:]) *\
                         samples[related_indices[0][i]][0][agent.index_][0, 0, ...]) == 0)
                if valid:
                    valid_prop_q_spvs.append(q_spvs[related_indices[0][i], j])
                else:
                    invalid_prop_q_spvs.append(q_spvs[related_indices[0][i], j])
                
        print('valid msg q_spvs:', np.mean(valid_msg_q_spvs))
        print('invalid msg q_spvs:', np.mean(invalid_msg_q_spvs))
        print('valid prop q_spvs:', np.mean(valid_prop_q_spvs))
        print('invalid prop q_spvs:', np.mean(invalid_prop_q_spvs))
        right = np.zeros(2)
        diffs1 = []
        diffs2 = []
        agent2_counter = np.zeros(3)
        agent2_counter2 = np.zeros(3)
        lengths = np.zeros(agent.training_config_.batch_size)
        last_act = np.zeros(2)
        all_msgs_taken = set()
        all_props_taken = set()
        for i in range(agent.training_config_.batch_size):
            lengths[i] = len(samples[i])
            last_act_idx = int(np.nonzero(samples[i][-1][0][1])[-1])
            last_act[int(last_act_idx >= agent.calendar_.num_msgs_)] += 1
            calendar_id = int(''.join(str(samples[i][0][agent.index_][0, 0, :])[1: -1].split()), 2)
            valid_msgs = agent.calendar_.get_msg(calendar_id)[0]
            valid_msg_idx = [agent.calendar_.all_msgs_.index(vm) for vm in valid_msgs]
            valid_mask = np.zeros(agent.calendar_.num_msgs_)
            slot_mask = np.squeeze(1 - samples[i][0][agent.index_])
            np.put(valid_mask, valid_msg_idx, 1)
            if np.sum(valid_mask) == agent.calendar_.num_msgs_:
                continue
            valid_msg_average_q = np.sum(q_predict[i, (samples[i][0][4] + agent.index_) % 2, 0: agent.calendar_.num_msgs_] * valid_mask) / np.sum(valid_mask)
            valid_propose_average_q = np.sum(q_predict[i, (samples[i][0][4] + agent.index_) % 2, agent.calendar_.num_msgs_: -1] * slot_mask) / np.sum(slot_mask)
            invalid_valid_propose_average_q = np.sum(q_predict[i, (samples[i][0][4] + agent.index_) % 2, agent.calendar_.num_msgs_: -1] * (1 - slot_mask)) / np.sum(1 - slot_mask)
            invalid_msg_average_q = np.sum(q_predict[i, (samples[i][0][4] + agent.index_) % 2, 0: agent.calendar_.num_msgs_] * (1 - valid_mask)) / np.sum(1 - valid_mask)
            #diffs1.append(valid_msg_average_q - valid_propose_average_q)
            diffs1.append(valid_propose_average_q - invalid_valid_propose_average_q)
            diffs2.append(valid_msg_average_q - invalid_msg_average_q)
            if valid_propose_average_q > invalid_valid_propose_average_q:
                right[0] += 1
            if valid_msg_average_q > invalid_msg_average_q:
                right[1] += 1
            #check helping agents actions
            other_turn = int(1 - (samples[i][0][4] + agent.index_) % 2)
            if len(samples[i]) - 1 > other_turn:
                agent2_counter[2] += 1
                action_idx = int(np.nonzero(samples[i][1 + other_turn][0][1])[-1])
                if action_idx >= agent.calendar_.num_msgs_:
                    all_props_taken.add(action_idx)
                    agent2_counter[1] += 1
                    valid_self = (np.sum((1 - samples[i][1 + other_turn][0][0][0, 0, agent.calendar_.num_slots_: -1]) * samples[i][0][1 - agent.index_][0, 0, ...]) == 0)
                    agent2_counter[0] += valid_self
                else:
                    all_msgs_taken.add(action_idx)
                    agent2_counter2[2] += 1
                    valid_self = (np.sum((samples[i][0][1 - agent.index_][0, 0, ...] - samples[i][1 + other_turn][0][0][0, 0, 0: agent.calendar_.num_slots_]) == -1) == 0)
                    agent2_counter2[1] += valid_self
                    msg_indices = np.where(agent.calendar_.action_tensor_[action_idx, :][0: agent.calendar_.num_slots_])
                    left_slot = samples[i][0][1 - agent.index_][0, 0, ...][msg_indices[0][0] - 1] if msg_indices[0][0] > 0 else 0
                    right_slot = samples[i][0][1 - agent.index_][0, 0, ...][msg_indices[0][-1] + 1] if msg_indices[0][-1] < agent.calendar_.num_slots_ - 1 else 0
                    agent2_counter2[0] += (left_slot + right_slot == 0) * valid_self
                    #if left_slot + right_slot != 0:
                    #    print('(short)', action_idx, samples[i][0][1 - agent.index_][0, 0, ...])
                    #else:
                    #    print('(long)', action_idx, samples[i][0][1 - agent.index_][0, 0, ...])
        print('agent helping takes %d different messages, %d different proposals' % (len(all_msgs_taken), len(all_props_taken)))
        print('agent helping propose: %f, agent helping correct propose: %f' % (agent2_counter[1] / agent2_counter[2],
                                                                                agent2_counter[0] / agent2_counter[1]))
        print('agent helping msg: %f, agent helping correct msg: %f, longest interval: %f' %
              (agent2_counter2[2] / agent2_counter[2],
               agent2_counter2[1] / agent2_counter2[2],
               agent2_counter2[0] / agent2_counter2[1]))
        print('batch average length: %f, end with msg: %f, end with propose: %f' % (np.mean(lengths),
                                                                                    last_act[0] / agent.training_config_.batch_size,
                                                                                    last_act[1] / agent.training_config_.batch_size))
        print('<%d: %d>batch valid propose > invalid propose ratio: %f, mean diff: %f, valid msg > invalid msg ratio: %f, mean diff: %f'\
              % (global_step, agent.index_, right[0] / agent.training_config_.batch_size, np.mean(diffs1),
                 right[1] / agent.training_config_.batch_size, np.mean(diffs2)))
    return q_learning_loss, policy_loss, belief_predict_loss

def main():
    from easydict import EasyDict as edict
    configs = [edict({'batch_size': 32,
                      'q_learning_rate': 1e-3,
                      'q_learning_gamma': 0.7,
                      'policy_learning_rate': 1e-3,
                      'memory_size': 1e3}),
               edict({'private_coding_length': 10,
                      'state_coding_length': 10,
                      'action_coding_length': 10,
                      'num_actions': 10,
                      'num_dishes': 4,
                      'acting_boltzman_beta': 10,
                      'fixed_fc_layer_info': [20, 10],
                      'fixed_single_length': [20, 10],
                      'fixed_context_length': [8, 6]}),
               edict({'history_dim': 10, 'fc_layer_info': [20, 15, 10]}),
               edict({'history_dim': 10, 'fc_layer_info': [20, 15, 10], 'num_actions': 10})]
    import copy
    sess = tf.Session()
    new_configs = copy.deepcopy(configs)
    agentA = Agent(sess, configs, 'A', True, False, True)
    agentB = Agent(sess, configs, 'B', False, True, True)

    init = tf.global_variables_initializer()
    sess.run(init)

    game = Kitchen(agentA.name_, agentB.name_, 0, -1, 1, np.arange(configs[1].num_actions), configs[1].num_dishes, 5)

    for _ in range(100):
        sar_sequence = play(agentA, agentB, game, 0.1)
        agentA.collect(sar_sequence)
        agentB.collect(sar_sequence)
    
    # recover_belief(agentB_infer, agentA.replay_buffer_[0: 10])
    train_step(agentA)
    train_step(agentB)
    return

if __name__ == '__main__':
    main()
