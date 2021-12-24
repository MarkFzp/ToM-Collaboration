import copy
import numpy as np
import tensorflow as tf

from Agent.agent import Agent
from Game.kitchen import Kitchen

import pdb

def play(agentA, agentB, game, epsilons, is_train = True):
    game.reset()
    initial_state = game.start()
    next_state = copy.deepcopy(initial_state)
    actor = agentA.name_
    agent_last_value_states = {agentA.name_: np.zeros([1, agentA.value_config_.history_dim]),
                               agentB.name_: np.zeros([1, agentB.value_config_.history_dim])}
    agentA_last_belief_states = np.zeros([1, agentA.belief_config_.history_dim])
    agentB_last_policy_states = np.zeros([1, agentB.policy_config_.history_dim])
    agentB_belief = np.ones(game.num_dishes_)
    agentB_belief /= game.num_dishes_
    last_action = -1
    last_belief_symbol = -1
    last_action_embedding = np.zeros([1, 1, agentA.game_config_.num_actions])
    sar_embedding_sequence = [(np.expand_dims(np.expand_dims(next_state[actor].observation.workplace_embed, 0), 0),
                               last_action_embedding, next_state[actor].observation.menu_embed, next_state[actor].private)]
   
    while not next_state[actor].terminate:
        if actor == agentA.name_:
            game_state_cache_A = {'value_history_state_encoding': agent_last_value_states[actor],
                                  'belief_history_state_encoding': agentA_last_belief_states,
                                  'last_action': last_action_embedding,
                                  'menu': np.expand_dims(next_state[agentA.name_]['observation'].menu_embed, 0),
                                  'target_encoding': np.expand_dims(np.expand_dims(initial_state[agentA.name_]['private'], 0), 0),
                                  'observation_encoding': np.expand_dims(np.expand_dims(next_state[agentA.name_]['observation'].workplace_embed, 0), 0)}
            last_action, agent_last_value_states[actor], _, agentA_last_belief_states, action_qs, belief_estimate\
                 = agentA.play(game_state_cache_A, epsilons[0])
            agent_last_value_states[agentB.name_], agentB_last_policy_states, _, likelihood =\
                agentB.look({'value_history_state_encoding': agent_last_value_states[agentB.name_],
                            'policy_history_state_encoding': agentB_last_policy_states,
                            'menu': np.expand_dims(next_state[agentB.name_]['observation'].menu_embed, 0),
                            'last_action': last_action_embedding,
                            'observation_encoding': np.expand_dims(np.expand_dims(next_state[agentB.name_]['observation'].workplace_embed, 0), 0)},
                            last_action)
            agentB_belief *= likelihood
            agentB_belief /= np.sum(agentB_belief)
            last_belief_symbol = np.random.choice(agentB.game_config_.num_dishes, 1, p = agentB_belief)[0]
            #last_belief_symbol = np.argmax(agentB_belief)
        else:
            game_state_cache_B = {'value_history_state_encoding': agent_last_value_states[actor],
                                  'policy_history_state_encoding': agentB_last_policy_states,
                                  'last_action': last_action_embedding,
                                  'menu': np.expand_dims(next_state[agentB.name_]['observation'].menu_embed, 0),
                                  'target_belief': agentB_belief,
                                  'observation_encoding': np.expand_dims(np.expand_dims(next_state[actor].observation.workplace_embed, 0), 0)}
            last_action, agent_last_value_states[actor], agentB_last_policy_states, _, action_qs, _\
                = agentB.play(game_state_cache_B, epsilons[1])
            agent_last_value_states[agentA.name_], _, agentA_last_belief_states, _ =\
                agentA.look({'value_history_state_encoding': agent_last_value_states[agentA.name_],
                             'belief_history_state_encoding': agentA_last_belief_states,
                             'observation_encoding': np.expand_dims(np.expand_dims(next_state[agentA.name_]['observation'].workplace_embed, 0), 0),
                             'last_action': last_action_embedding}, last_action)
            last_belief_symbol = -1
        last_action_embedding = np.zeros([1, 1, agentB.game_config_.num_actions])
        np.put(last_action_embedding, last_action, 1)
        last_belief_embedding = np.zeros([1, 1, agentA.game_config_.num_dishes])
        if last_belief_symbol != -1:
            np.put(last_belief_embedding, last_belief_symbol, 1)
        next_state = game.proceed({actor: last_action})
        if is_train:
            sar_embedding_sequence.append((np.expand_dims(np.expand_dims(next_state[actor].observation.workplace_embed, 0), 0),
                                           last_action_embedding, next_state[actor].reward, last_belief_embedding))
        else:
            sar_embedding_sequence.append((np.expand_dims(np.expand_dims(next_state[actor].observation.workplace_embed, 0), 0),
                                        last_action_embedding, next_state[actor].reward,
                                        copy.deepcopy(agentB_belief), action_qs, likelihood, belief_estimate))
        actor = agentA.name_ if actor == agentB.name_ else agentB.name_
    #[(empty-state, Null-action, menu, order), (s, a, r, m), (s, a, r, m), ...]
    return sar_embedding_sequence

def train_step(agent, global_step, switch = 5e4):
    samples_idx = np.random.choice(len(agent.replay_buffer_), agent.training_config_.batch_size, replace = False)
    samples = [agent.replay_buffer_[i] for i in samples_idx]
    samples_length = [len(agent.replay_buffer_[i]) for i in samples_idx]
    max_length = np.max(samples_length)
    
    previous_actions = np.zeros([agent.training_config_.batch_size, max_length,
                                 agent.game_config_.action_coding_length])
    previous_states = np.zeros([agent.training_config_.batch_size, max_length, agent.game_config_.state_coding_length])
    policy_rnn_states = np.zeros([agent.training_config_.batch_size, agent.policy_config_.history_dim])
    value_rnn_states = np.zeros([agent.training_config_.batch_size, agent.value_config_.history_dim])
    belief_rnn_states = np.zeros([agent.training_config_.batch_size, agent.belief_config_.history_dim])
    menu_embedding = np.zeros([agent.training_config_.batch_size,
                               agent.game_config_.num_dishes,
                               agent.game_config_.private_coding_length])
    goal_embedding = np.zeros([agent.training_config_.batch_size, max_length, agent.game_config_.num_dishes])
    for i in range(len(samples)):
        for si in range(len(samples[i])):
            goal_embedding[i, si, :] = samples[i][0][3]

    action_masks = np.zeros([agent.training_config_.batch_size, max_length])
    action_spvs = []
    
    for i in range(len(samples)):
        menu_embedding[i, ...] = np.expand_dims(samples[i][0][2], 0)
        for si in range(1, len(samples[i])):
            previous_actions[i, si, :] = samples[i][si][1]
            previous_states[i, si, :] = samples[i][si][0]
        for si in range(len(samples[i]) - 1):
            if si % 2 != agent.index_:
                action_masks[i, si] = 1
                action_spvs.append(samples[i][si + 1][1])
    action_masks = np.expand_dims(np.nonzero(np.reshape(action_masks,
                                                        [agent.training_config_.batch_size * max_length, 1]))[0], 1)
    action_spvs = np.squeeze(np.array(action_spvs))
    q_spvs = np.zeros([agent.training_config_.batch_size, max_length])
    q_spvs_mask = np.zeros([agent.training_config_.batch_size, max_length, agent.game_config_.num_actions])
    belief_mask = np.zeros([agent.training_config_.batch_size, max_length])
    belief_spvs = []
    
    feed_dict = {agent.policy_network_.history_state_input_: policy_rnn_states,
                 agent.q_target_.history_state_input_: value_rnn_states,
                 agent.history_encoding_: previous_actions,
                 agent.state_encoding_: previous_states,
                 agent.fixed_settings_: menu_embedding,
                 agent.policy_spvs_mask_: action_masks,
                 agent.policy_spvs_: action_spvs}
    
    if agent.index_ == 0:
        feed_dict[agent.private_belief_self_] = goal_embedding
        feed_dict[agent.belief_network_.history_state_input_] = belief_rnn_states
    else:
        feed_dict[agent.private_belief_other_] = goal_embedding
    _, policy_loss, action_target_qs, other_policy_est =\
        agent.sess_.run([agent.policy_training_op_, agent.policy_loss_,
                         agent.q_target_.values_, agent.policy_network_.action_prob_], feed_dict) 
    
    max_target_qs = np.max(action_target_qs, axis = 2)
    for i in range(len(samples)):
        for si, step in enumerate(samples[i][0: -1]):
            if si % 2 == agent.index_:
                next_action_index = np.nonzero(samples[i][si + 1][1])[2][0]
                q_spvs_mask[i, si, next_action_index] = 1
                reward1 = samples[i][si + 1][2]
                reward2 = samples[i][si + 2][2] if si < len(samples[i]) - 2 else 0
                next_q = max_target_qs[i, si + 2] if si < len(samples[i]) - 3 else 0
                q_spvs[i, si] = reward1 + agent.training_config_.q_learning_gamma *\
                                (reward2 + agent.training_config_.q_learning_gamma * next_q)
                belief_mask[i, si] = 1
                belief_spvs.append(samples[i][si + 1][3])

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
    if agent.index_ == 0:
        feed_dict[agent.belief_spvs_mask_1_] = q_spvs_mask
        feed_dict[agent.belief_spvs_mask_2_] = belief_mask
        feed_dict[agent.belief_spvs_] = np.squeeze(np.array(belief_spvs))
        if global_step > 2 * switch:
            belief_train_op = agent.belief_training_op_
            belief_loss_op = agent.belief_loss_
            likelihood_op = agent.belief_network_.likelihood_pre_

    _, _, q_predict, q_learning_loss, belief_predict_loss, likelihood =\
        agent.sess_.run([agent.q_training_op_, belief_train_op,
                         agent.q_primary_.values_, agent.q_learning_loss_, belief_loss_op, likelihood_op], feed_dict)
    # if global_step % 50 == 0:
    #     pdb.set_trace()
    if global_step % 1000 == 0:
        right = 0
        diffs = []
        # if q_learning_loss < 0.04:
        #     print(np.sum(q_predict * q_spvs_mask, axis = 2))
        #     print(q_spvs)
        for i in range(agent.training_config_.batch_size):
            all_included = samples[i][0][2][int(np.where(samples[i][0][3])[0])]
            last_q_idx = -2 - (len(samples[i]) % 2 != agent.index_)
            have = samples[i][max(-1 * len(samples[i]), last_q_idx - 1)][0]
            needed = all_included - have[0, 0, :] > 0
            correct_average = np.sum(q_predict[i, last_q_idx, :] * needed) / np.sum(needed)
            wrong_average = np.sum(q_predict[i, last_q_idx, :] * (1 - needed)) / np.sum(1 - needed)
            diffs.append(correct_average - wrong_average)
            if correct_average > wrong_average:
                right += 1
            # print('correct: %f, wrong: %f' % (correct_average, wrong_average))
        print('<%d: %d>batch correct > wrong ratio: %f, mean diff: %f'\
              % (global_step, agent.index_, 1.0 * right / agent.training_config_.batch_size, np.mean(diffs)))
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