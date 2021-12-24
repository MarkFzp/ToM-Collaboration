import tensorflow as tf
import numpy as np
from qnet import QNet
from policy import Policy
from replaybuffer import ReplayBuffer
import os
import sys
if len(sys.argv) == 1:
    from config import config
else:
    exec("from {} import config".format(sys.argv[1]))
sys.path.append("../../..")
from Game.kitchen import Kitchen
from util import change_pair_test
from copy import deepcopy
from shutil import copyfile

# for debugging
import pdb
from pprint import pprint
from collections import OrderedDict
np.set_printoptions(precision=4, suppress=True)


def buffer_sample_experience(config, player1, player2, kitchen, policy1, policy2, buffer, count):
    for _ in range(count):
        kitchen.reset(train = not config.test)
        start_ret = kitchen.start()
        p1_start_ret = start_ret[player1]
        p2_start_ret = start_ret[player2]
        menu = p1_start_ret['observation'].menu_embed
        menu_len = len(menu)
        # if null_action_one_hot in menu:
        #     print(menu)
        #     raise Exception('null dish in menu')
        # zero padding to menu
        if menu_len != config.num_candidate:
            print('found short menu!!!!!')
            assert(menu_len < config.num_candidate)
            menu = np.stack([menu, np.zeros([config.num_candidate - menu_len, config.num_ingredient])], axis = 0)
        menu = menu.reshape([-1])

        p1_target = deepcopy(p1_start_ret['private'])
        p1_menu_and_target = np.concatenate([menu, p1_target])
        p2_menu_and_target = np.concatenate([menu, p2_target])


        # player 1 acts first
        # player 1 & 2 act turn by turn
        terminate = False
        t = 0
        in_state = [None, None]
        prev_action = [-1, -1]
        prepared_ingred = np.zeros([config.num_ingredient])
        
        trajectory = {'menu': menu, 'p1_target': p1_target, 'sarsa': [], 'success': None}
        
        while not terminate:
            p1_obs = np.concatenate([p1_menu_and_target, prepared_ingred])
            p1_in_action = prev_action[0]
            p1_in_state = in_state[0]
            p1_out_action, p1_out_action_prob, p1_out_state = policy1.single_sample_act(p1_obs, p1_in_action, player1, p1_in_state)
            p2_obs = np.concatenate([p2_menu_and_target, prepared_ingred])
            p2_in_action = prev_action[1]
            p2_in_state = in_state[1]
            p2_out_action, p2_out_action_prob, p2_out_state = policy2.single_sample_act(p2_obs, p2_in_action, player2, p2_in_state)
            # pdb.set_trace()
            if t % 2 == 0:
                env_feedback = kitchen.proceed({player1: p1_out_action})[player1]
                new_prepared_ingred = env_feedback.observation.workplace_embed
                reward = env_feedback.reward
                terminate = env_feedback.terminate
                
                if terminate:
                    trajectory['success'] = env_feedback.success
                trajectory['sarsa'].append((prepared_ingred, p1_out_action, reward, p1_out_action_prob))
                
                in_state = [p1_out_state, p2_out_state]
                prev_action = [p1_out_action, -1]
                prepared_ingred = deepcopy(new_prepared_ingred)
            else:
                env_feedback = kitchen.proceed({player2: p2_out_action})[player2]
                new_prepared_ingred = env_feedback.observation.workplace_embed
                reward = env_feedback.reward
                terminate = env_feedback.terminate

                if terminate:
                    trajectory['success'] = env_feedback.success
                trajectory['sarsa'].append((prepared_ingred, p2_out_action, reward, p2_out_action_prob))
                
                in_state = [p1_out_state, p2_out_state]
                prev_action = [-1, p2_out_action]
                prepared_ingred = deepcopy(new_prepared_ingred)

            t += 1
        # if len(trajectory['sarsa']) > 1:
        #     pdb.set_trace()
        buffer.push(trajectory)




def fit_qnet(config, buffer, qnet_primary, qnet_target):
    all_p1_states = []
    all_p2_states = []
    all_player_idxs = []
    all_action_chosens = []
    all_action_probs = []
    all_td_lambda_values = []

    # to break correlation within a trajectory, sample some epoches from the trajectory
    sampled_experience = buffer.get(config.batch_size)
    for trajectory in sampled_experience:
        menu = trajectory['menu']
        p1_target = trajectory['p1_target']

        traj_len = len(trajectory['sarsa'])
        assert(traj_len >= 1)
        
        p1_states = []
        p2_states = []
        player_idxs = []
        action_chosens = []
        rewards = []
        action_probs = []
        # trajectory['sarsa'] is (prepared_ingred, out_action, reward, out_action_prob))
        for epoch, sarsa in enumerate(trajectory['sarsa']):
            prepared_ingred, action_chosen, reward, action_prob = sarsa
            p1_states.append(np.concatenate([menu, p1_target, prepared_ingred]))
            p2_states.append(np.concatenate([menu, p2_target, prepared_ingred]))
            player_idxs.append(epoch % 2)
            action_chosens.append(action_chosen)
            rewards.append(reward)
            action_probs.append(action_prob)
        
        
        p1_states = np.array(p1_states)
        p2_states = np.array(p2_states)
        player_idxs = np.array(player_idxs)
        action_chosens = np.array(action_chosens)
        action_probs = np.array(action_probs)
        # no need to collect target q at t = 0
        if traj_len >= 2:
            q_values_after_t0 = qnet_target.get_q(p1_states[1:], player_idxs[1:], action_chosens[1:])
            # q_values = qnet_target.get_q(p1_states, player_idxs, action_chosens)
        
        # y(s_t) = r_t + gamma * [lamda * y(s_t+1) + (1 - lamda) * Q(s_t+1)]
        td_lambda_values = [None] * traj_len
        for epoch in reversed(range(traj_len)):
            if epoch == traj_len - 1:
                td_lambda_values[epoch] = rewards[epoch]
            else:
                td_lambda_value = \
                    rewards[epoch] \
                    + \
                    config.gamma * ( \
                        config.td_lambda * td_lambda_values[epoch + 1] \
                        + \
                        (1 - config.td_lambda) * q_values_after_t0[epoch] \
                    )
                td_lambda_values[epoch] = td_lambda_value
        td_lambda_values = np.array(td_lambda_values)

        all_p1_states.append(p1_states)
        all_p2_states.append(p2_states)
        all_player_idxs.append(player_idxs)
        all_action_chosens.append(action_chosens)
        all_action_probs.append(action_probs)
        all_td_lambda_values.append(td_lambda_values)
    
    all_p1_states_np = np.concatenate(all_p1_states, axis = 0)
    all_player_idxs_np = np.concatenate(all_player_idxs, axis = 0)#[sample_epoch]
    all_action_chosens_np = np.concatenate(all_action_chosens, axis = 0)#[sample_epoch]
    all_td_lambda_values_np = np.concatenate(all_td_lambda_values, axis = 0)#[sample_epoch]
    
    if config.q_decorrelation:
        epoch_count = len(all_p1_states_np)
        sample_epoch = np.random.choice(epoch_count, size = round(config.batch_size / 5), replace = False)
        loss, _ = qnet_primary.fit_target_q(all_p1_states_np[sample_epoch], all_player_idxs_np[sample_epoch], all_action_chosens_np[sample_epoch], all_td_lambda_values_np[sample_epoch])
        q_chosen = None
    else:
        loss, q_chosen = qnet_primary.fit_target_q(all_p1_states_np, all_player_idxs_np, all_action_chosens_np, all_td_lambda_values_np)

    # sample_traj = np.random.choice(config.batch_size, size = config.batch_size, replace = False)
    # return loss, np.array(all_p1_states)[sample_traj].tolist(), \
    #              np.array(all_p2_states)[sample_traj].tolist(), \
    #              np.array(all_player_idxs)[sample_traj].tolist(), \
    #              np.array(all_action_chosens)[sample_traj].tolist(), \
    #              np.array(all_action_probs)[sample_traj].tolist(), \
    #              all_td_lambda_values
    return loss, all_p1_states, all_p2_states, all_player_idxs, all_action_chosens, all_action_probs, all_td_lambda_values, q_chosen




def policy_grad(player1, player2, p1_states, p2_states, player_idxs, action_chosens, action_probs, qnet_primary, policy):
    assert(len(p1_states) == len(p2_states) == len(player_idxs) == len(action_chosens) == len(action_probs))
    
    traj_max_len = 0
    for traj in p1_states:
        if traj_max_len < len(traj):
            traj_max_len = len(traj)
    
    advs = []
    log_prob_idxs = []
    in_xs = []
    all_action_qs = []

    for traj_idx, traj in enumerate(zip(p1_states, p2_states, player_idxs, action_chosens, action_probs)):
        p1_state, p2_state, player_idx, action_chosen, action_prob = traj
        assert(len(p1_state) == len(p2_state) == len(player_idx) == len(action_chosen) == len(action_prob))
        
        traj_len = len(p1_state)
        all_action_q_value = qnet_primary.get_q(
            np.concatenate([p1_state] * config.num_action, axis = 1).reshape([-1, config.q_state_dim]),
            np.transpose(np.stack([player_idx] * config.num_action, axis = 0)).reshape([-1]),
            np.concatenate([all_action] * traj_len, axis = 0)
        )
        all_action_q_value = all_action_q_value.reshape([traj_len, config.num_action])
        action_chosen_q = all_action_q_value[range(traj_len), action_chosen]
        baseline = np.sum(all_action_q_value * action_prob, axis = 1)
        adv = action_chosen_q - baseline

        all_action_qs.append(all_action_q_value)
        advs.extend(adv)

        prev_action = np.concatenate([[-1], action_chosen[:-1]], axis = 0)
        in_x_p1 = []
        in_x_p2 = []
        for t in range(traj_len):
            if t == 0:
                in_x_p1.append(np.concatenate([p1_state[t], null_action_one_hot, one_hot_p_idx[player1]], axis = 0))
                in_x_p2.append(np.concatenate([p2_state[t], null_action_one_hot, one_hot_p_idx[player2]], axis = 0))
                log_prob_idxs.append([traj_idx * 2, t, action_chosen[t]])
            elif t % 2 == 0:
                in_x_p1.append(np.concatenate([p1_state[t], null_action_one_hot, one_hot_p_idx[player1]], axis = 0))
                in_x_p2.append(np.concatenate([p2_state[t], one_hot_action[prev_action[t]], one_hot_p_idx[player2]], axis = 0))
                log_prob_idxs.append([traj_idx * 2, t, action_chosen[t]])
            else:
                in_x_p1.append(np.concatenate([p1_state[t], one_hot_action[prev_action[t]], one_hot_p_idx[player1]], axis = 0))
                in_x_p2.append(np.concatenate([p2_state[t], null_action_one_hot, one_hot_p_idx[player2]], axis = 0))
                log_prob_idxs.append([traj_idx * 2 + 1, t, action_chosen[t]])

        in_x_p1 = np.stack(in_x_p1, axis = 0)
        in_x_p2 = np.stack(in_x_p2, axis = 0)

        if traj_len < traj_max_len:
            zero_padding = np.zeros([traj_max_len - traj_len, config.in_x_dim])
            in_x_p1 = np.concatenate([in_x_p1, zero_padding], axis = 0)
            in_x_p2 = np.concatenate([in_x_p2, zero_padding], axis = 0)
        
        in_xs.append(in_x_p1)
        in_xs.append(in_x_p2)

    in_xs = np.stack(in_xs, axis = 0)
    log_prob_idxs = np.array(log_prob_idxs)
    advs = np.array(advs)

    loss, epsilon = policy.train(in_xs, log_prob_idxs, advs)

    return loss, epsilon, advs, all_action_qs



def policy_grad_separate(player1, player2, p1_states, p2_states, player_idxs, action_chosens, action_probs, qnet_primary, policy1, policy2):
    assert(len(p1_states) == len(p2_states) == len(player_idxs) == len(action_chosens) == len(action_probs))
    
    traj_max_len = 0
    for traj in p1_states:
        if traj_max_len < len(traj):
            traj_max_len = len(traj)
    
    advs_1 = []
    advs_2 = []
    advs = []
    log_prob_idxs_1 = []
    log_prob_idxs_2 = []
    in_xs_1 = []
    in_xs_2 = []
    all_action_qs = []

    for traj_idx, traj in enumerate(zip(p1_states, p2_states, player_idxs, action_chosens, action_probs)):
        p1_state, p2_state, player_idx, action_chosen, action_prob = traj
        assert(len(p1_state) == len(p2_state) == len(player_idx) == len(action_chosen) == len(action_prob))
        
        traj_len = len(p1_state)
        all_action_q_value = qnet_primary.get_q(
            np.concatenate([p1_state] * config.num_action, axis = 1).reshape([-1, config.q_state_dim]),
            np.transpose(np.stack([player_idx] * config.num_action, axis = 0)).reshape([-1]),
            np.concatenate([all_action] * traj_len, axis = 0)
        )
        all_action_q_value = all_action_q_value.reshape([traj_len, config.num_action])
        action_chosen_q = all_action_q_value[range(traj_len), action_chosen]
        baseline = np.sum(all_action_q_value * action_prob, axis = 1)
        adv = action_chosen_q - baseline

        all_action_qs.append(all_action_q_value)
        advs_1.extend(adv[0::2])
        advs_2.extend(adv[1::2])
        advs.extend(adv)

        prev_action = np.concatenate([[-1], action_chosen[:-1]], axis = 0)
        in_x_p1 = []
        in_x_p2 = []
        for t in range(traj_len):
            if t == 0:
                in_x_p1.append(np.concatenate([p1_state[t], null_action_one_hot, one_hot_p_idx[player1]], axis = 0))
                in_x_p2.append(np.concatenate([p2_state[t], null_action_one_hot, one_hot_p_idx[player2]], axis = 0))
                log_prob_idxs_1.append([traj_idx, t, action_chosen[t]])
            elif t % 2 == 0:
                in_x_p1.append(np.concatenate([p1_state[t], null_action_one_hot, one_hot_p_idx[player1]], axis = 0))
                in_x_p2.append(np.concatenate([p2_state[t], one_hot_action[prev_action[t]], one_hot_p_idx[player2]], axis = 0))
                log_prob_idxs_1.append([traj_idx, t, action_chosen[t]])
            else:
                in_x_p1.append(np.concatenate([p1_state[t], one_hot_action[prev_action[t]], one_hot_p_idx[player1]], axis = 0))
                in_x_p2.append(np.concatenate([p2_state[t], null_action_one_hot, one_hot_p_idx[player2]], axis = 0))
                log_prob_idxs_2.append([traj_idx, t, action_chosen[t]])

        in_x_p1 = np.stack(in_x_p1, axis = 0)
        in_x_p2 = np.stack(in_x_p2, axis = 0)

        if traj_len < traj_max_len:
            zero_padding = np.zeros([traj_max_len - traj_len, config.in_x_dim])
            in_x_p1 = np.concatenate([in_x_p1, zero_padding], axis = 0)
            in_x_p2 = np.concatenate([in_x_p2, zero_padding], axis = 0)
        
        in_xs_1.append(in_x_p1)
        in_xs_2.append(in_x_p2)

    in_xs_1 = np.stack(in_xs_1, axis = 0)
    log_prob_idxs_1 = np.array(log_prob_idxs_1)
    advs_1 = np.array(advs_1)

    in_xs_2 = np.stack(in_xs_2, axis = 0)
    log_prob_idxs_2 = np.array(log_prob_idxs_2)
    advs_2 = np.array(advs_2)

    loss_1, epsilon_1 = policy1.train(in_xs_1, log_prob_idxs_1, advs_1)
    loss_2, epsilon_2 = policy2.train(in_xs_2, log_prob_idxs_2, advs_2)
    mean_loss = (loss_1 + loss_2) / 2
    assert(epsilon_1 == epsilon_2)

    return mean_loss, epsilon_1, advs, all_action_qs


def unique_id(menu, target_id, action_0, config):
    menu_2d = menu.reshape([config.num_candidate, config.num_ingredient])
    other_context = np.sum(np.delete(menu_2d, target_id, axis = 0), axis = 0)
    target_one_hot = menu_2d[target_id]
    target_ingred = np.where(target_one_hot)[0]
    target_ingred_is_unique_id = np.logical_not(other_context[target_ingred])
    unique_id_exist = True in target_ingred_is_unique_id
    if not unique_id_exist:
        return False, False
    elif action_0 in target_ingred and target_ingred_is_unique_id[np.where(target_ingred == action_0)[0][0]]:
        return True, True
    else:
        return True, False
    
    


def brint(debug_info, config):
    adv_pointer = 0
    trajs, advs, q_chosens = debug_info

    traj_count = 0
    menu_with_unique_count = 0
    send_unique_count = 0
    succ_after_send_unique_count = 0
    succ_not_send_unique_count = 0

    for traj in trajs:
        p1_state, p2_state, action_chosen, action_prob, td_lambda, all_action_q = traj
        traj_len = len(p1_state)
        assert(traj_len == len(p2_state) == len(action_chosen) == len(action_prob) == len(td_lambda) == len(all_action_q))
        
        menu = p1_state[0, :config.num_candidate * config.num_ingredient]
        action_0 = action_chosen[0]

        assert(np.mean(menu == p1_state[:, :config.num_candidate * config.num_ingredient]) == 1)
        assert(np.mean(menu == p2_state[:, :config.num_candidate * config.num_ingredient]) == 1)

        p1_target = p1_state[0, config.num_candidate * config.num_ingredient: config.num_candidate * config.num_ingredient + config.num_candidate]
        p2_target = p2_state[0, config.num_candidate * config.num_ingredient: config.num_candidate * config.num_ingredient + config.num_candidate]
        assert(np.mean(p1_target == p1_state[:, config.num_candidate * config.num_ingredient: config.num_candidate * config.num_ingredient + config.num_candidate]) == 1)
        assert(np.mean(p2_target == p2_state[:, config.num_candidate * config.num_ingredient: config.num_candidate * config.num_ingredient + config.num_candidate]) == 1)

        target_id = np.where(p1_target)[0][0]

        action_one_hot = one_hot_action[action_chosen]
        adv = advs[adv_pointer: adv_pointer + traj_len]
        q_chosen = q_chosens[adv_pointer: adv_pointer + traj_len]
        ingred = p1_state[:, -config.num_ingredient:]
        assert(np.mean(ingred == p2_state[:, -config.num_ingredient:]) == 1)

        unique_id_exist, unique_id_chosen = unique_id(menu, target_id, action_0, config)
        traj_count += 1
        menu_with_unique_count += unique_id_exist
        send_unique_count += unique_id_chosen
        succ = True if td_lambda[-1] == config.succeed_reward else False
        if unique_id_exist and unique_id_chosen:
            succ_after_send_unique_count += succ
        elif unique_id_exist and not unique_id_chosen:
            succ_not_send_unique_count += succ

        if config.verbose:
            pprint([
                ('menu', menu.reshape((config.num_candidate, config.num_attribute)).tolist()), 
                ('p1_target', p1_target.tolist()), 
                ('p2_target', p2_target.tolist()), 
                ('ingred', ingred.tolist()), 
                ('action', action_one_hot.tolist()),
                ('action_prob', action_prob), 
                ('td', td_lambda), 
                ('all_q', all_action_q), 
                ('q', q_chosen), 
                ('adv', adv),
                ('unique_id_exist', unique_id_exist),
                ('unique_id_chosen', unique_id_chosen)
            ])
            cmd = input()
            if cmd == ' ':
                return
        
        adv_pointer += traj_len
    
    pprint([
        ('traj_count', traj_count),
        ('menu_with_unique_count', menu_with_unique_count), 
        ('send_unique_count', send_unique_count),
        ('succ_after_send_unique_count', succ_after_send_unique_count),
        ('succ_not_send_unique_count', succ_not_send_unique_count)
    ])
    


def main():
    if not config.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    sess_config = tf.ConfigProto(allow_soft_placement = True, log_device_placement = False)
    sess_config.gpu_options.allow_growth = True
    sess = tf.Session(config = sess_config)

    # DONT change players name
    player1 = 0 # teacher
    player2 = 1 # student
    kitchen = Kitchen(player1, player2, \
                    config.step_reward, config.fail_reward, config.succeed_reward, \
                    range(config.num_attribute), config.num_candidate, config.max_attribute, \
                    dataset_path = config.dataset_path)
    qnet_primary = QNet(config, sess, False)
    qnet_target = QNet(config, sess, True)
    
    buffer = ReplayBuffer(config)

    if config.change_pair_test or config.separate_policy:
        policy1 = Policy(config, sess, config.policy_name_appendix)
        policy2 = Policy(config, sess, config.policy_name_appendix_2)
    else:
        policy = Policy(config, sess, config.policy_name_appendix)
        policy1 = policy2 = policy
    
    if config.change_pair_test:
        ''' 
        if cannot load, check the name of gru!!!
        '''
        init = tf.global_variables_initializer()
        sess.run(init)
        saver_1 = tf.train.Saver(var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope = 'policy/'))

        if os.path.exists(config.ckpt_path_2 + '.index'):
            saver_1.restore(sess, config.ckpt_path_2)
        else:
           print('ckpt_path does not exist', config.ckpt_path_2)
           exit()
        
        p1_var = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope = "policy/")
        p2_var = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope = "policy_2/")
        update_op = tf.group([v_t.assign(v) for v_t, v in zip(p2_var, p1_var)])
        sess.run(update_op)

        if os.path.exists(config.ckpt_path + '.index'):
            saver_1.restore(sess, config.ckpt_path)
        else:
           print('ckpt_path does not exist', config.ckpt_path)
           exit()


        # policy_name_1 = 'policy/' if config.policy_name_appendix is None else 'policy_{}/'.format(config.policy_name_appendix)
        # policy_name_2 = 'policy/' if config.policy_name_appendix_2 is None else 'policy_{}/'.format(config.policy_name_appendix_2)
        
        # saver_1 = tf.train.Saver(var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope = policy_name_1))
        # saver_2 = tf.train.Saver(var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope = policy_name_2))
        # if os.path.exists(config.ckpt_path + '.index'):
        #     saver_1.restore(sess, config.ckpt_path)
        # else:
        #     print('ckpt_path does not exist', config.ckpt_path)
        #     exit()
        # if os.path.exists(config.ckpt_path_2 + '.index'):
        #     saver_2.restore(sess, config.ckpt_path_2)
        # else:
        #     print('ckpt_path does not exist', config.ckpt_path_2)
        #     exit()
            
        change_pair_test(config, player1, player2, kitchen, policy1, policy2, buffer, config.test_batch_size)
        exit()
    
    # init = tf.global_variables_initializer()
    # sess.run(init)

    saver = tf.train.Saver()
    # policy2 = Policy(config, sess, '2')
    # saver = tf.train.Saver(var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope = 'policy/'))
    
    ckpt_name = os.path.splitext(os.path.basename(config.ckpt_path))[0]

    if os.path.exists(config.load_path + '.index'):
        saver.restore(sess, config.load_path)
    else:
        print('load_path does not exist')
        init = tf.global_variables_initializer()
        sess.run(init)
        config_name = 'config_' + ckpt_name + '.py'
        # if not os.path.exists(config_name):
        copyfile('config.py', config_name)

    # saver_2 = tf.train.Saver(var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope = 'policy'))
    # p1_var = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope = "policy/")
    # p2_var = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope = "policy_2/")
    # update_op = tf.group([v_t.assign(v) for v_t, v in zip(p2_var, p1_var)])
    # sess.run(update_op)
    # saver_2.save(sess, '2_' + config.ckpt_path)
    # exit()


    # Get all the variables in the Q primary network.
    q_primary_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope = "primary")
    # Get all the variables in the Q target network.
    q_target_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope = "target")

    # to initialize target QNet to primary QNet
    qnet_target.set_target_net_update_op(q_primary_vars, q_target_vars)
    qnet_target.target_net_update()
    
    global null_state
    null_state = np.zeros([config.q_state_dim], dtype = np.int32)
    global null_action_one_hot
    null_action_one_hot = np.zeros([config.num_action], dtype = np.int32)
    global all_action
    all_action = np.arange(config.num_action, dtype = np.int32)
    global p2_target
    if config.ad_hoc_structure:
        p2_target = np.ones([config.num_candidate]) / config.num_candidate
    elif config.p2_target_all_zero:
        p2_target = np.zeros([config.num_candidate], dtype = np.int32)
    else:
        p2_target = np.ones([config.num_candidate]) / config.num_candidate

    global one_hot_action
    one_hot_action = np.eye(config.num_action)
    global one_hot_p_idx
    one_hot_p_idx = np.eye(config.num_player)
    
    # buffer_sample_experience(config, player1, player2, kitchen, policy1, policy2, buffer, buffer.buffer_size)

    qnet_losses = []
    pg_losses = []
    accuracy_list = []
    p2_wrong_rate_list = []
    mean_traj_len_list = []

    test_acc = []

    
    for episode in range(0 if config.ctd_itr is None else config.ctd_itr, config.itr):
        if (episode + 1) % config.test_itr == 0:
            config.test = True
            inner_test_acc = []
            for i in range(config.test_batch_size // config.batch_size):
                buffer_sample_experience(config, player1, player2, kitchen, policy1, policy2, buffer, config.batch_size)
                acc, _ = buffer.accuracy()
                inner_test_acc.append(acc)
            batch_test_acc = np.mean(inner_test_acc)
            test_acc.append(batch_test_acc)
            print('[test acc]: {}'.format(batch_test_acc))
            print()
            config.test = False

        buffer_sample_experience(config, player1, player2, kitchen, policy1, policy2, buffer, config.batch_size)
        
        mean_traj_len_list.append(buffer.mean_traj_len())
        
        qnet_loss, p1_states, p2_states, player_idxs, action_chosens, action_probs, td_lambda_values, q_chosen \
            = fit_qnet(config, buffer, qnet_primary, qnet_target)
        
        qnet_losses.append(qnet_loss)

        
        if episode % config.qnet_target_update_itr == 0 and episode != 0:
            qnet_target.target_net_update()

        if config.separate_policy:
            pg_loss, epsilon, advs, all_action_qs = policy_grad_separate(player1, player2, p1_states, p2_states, \
                player_idxs, action_chosens, action_probs, qnet_primary, policy1, policy2)
        else:
            pg_loss, epsilon, advs, all_action_qs = policy_grad(player1, player2, p1_states, p2_states, \
                player_idxs, action_chosens, action_probs, qnet_primary, policy)

        pg_losses.append(pg_loss)
        
        acc, p2_wrong_rate = buffer.accuracy()

        accuracy_list.append(acc)
        p2_wrong_rate_list.append(p2_wrong_rate)

        if episode % config.print_itr == 0:
            print('[t{}]'.format(episode))
            print('accuracy: ', np.mean(accuracy_list))
            accuracy_list = []
            print('qnet loss: ', np.mean(qnet_losses))
            qnet_losses = []
            print('pg loss: ', np.mean(pg_losses))
            pg_losses = []
            print('p2_wrong: ', np.mean(p2_wrong_rate_list))
            p2_wrong_rate_list = []
            print('mean_traj_len: ', np.mean(mean_traj_len_list))
            mean_traj_len_list = []
            print('epsilon: ', epsilon)
            print()

            if config.debug:
                debug_info = (zip(p1_states, p2_states, action_chosens, action_probs, td_lambda_values, all_action_qs), advs, q_chosen)
                brint(debug_info, config)
                # pdb.set_trace()

        if (episode + 1) % config.save_itr == 0 and episode != 0:
            saver.save(sess, config.ckpt_path)
            print('ckpt saved')
            print()
    
    
    np.save(ckpt_name + '_test_acc', np.array(test_acc))

        


if __name__ == '__main__':
    main()
