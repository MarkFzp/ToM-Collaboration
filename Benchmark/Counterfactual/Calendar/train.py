import tensorflow as tf
import numpy as np
from qnet import QNet
from policy import Policy
from config import config
import os
import sys
sys.path.append("../../..")
from Game.calendar import Calendar
from util import change_pair_test, max_len_msg_test
from copy import deepcopy
from shutil import copyfile

# for debugging
# import pdb
# from pprint import pprint
# from collections import OrderedDict
np.set_printoptions(precision=4, suppress=True)


def sample_trajs():
    trajs = []
    max_traj_len = -1

    succeed = []
    msg_succeed = []
    action_succeed = []
    traj_len = []
    p2_wrong_count = 0

    for _ in range(config.batch_size if not config.test else config.test_batch_size):
        calendar.reset(train = not config.test)
        p1_calendar = calendar.start()[player1].private
        p2_calendar = calendar.start()[player2].private
        
        # player 1 & 2 act turn by turn
        terminate = False
        t = 0
        fst_act_player = np.random.randint(0, 2)
        
        in_state = [None, None]
        prev_action_p_idx = null_action_p_idx

        traj = {'p1_calendar': p1_calendar, 'p2_calendar': p2_calendar, 'fst_act_player': fst_act_player, \
            'reward': [], 'pi': [], 'action_p_idx': [], 'action': [], 'traj_len': None, \
            'in_x_1': [], 'in_x_2': []}
        
        while not terminate:
            if t >= config.terminate_len:
                succeed.append(False)
                if np.random.random() < 0.5:
                    p2_wrong_count += 1
                traj['traj_len'] = t
                traj_len.append(t)
                break

            else:
                in_x_1 = np.concatenate([p1_calendar, prev_action_p_idx, p1_idx_oh])
                in_x_2 = np.concatenate([p2_calendar, prev_action_p_idx, p2_idx_oh])
                action_1, action_prob_1, state_1 = policy.sample_action(in_x_1, in_state[0])
                action_2, action_prob_2, state_2 = policy.sample_action(in_x_2, in_state[1])
                in_state = [state_1, state_2]
                traj['in_x_1'].append(in_x_1)
                traj['in_x_2'].append(in_x_2)

                if t % 2 == fst_act_player:
                    env_feedback = calendar.proceed({player1: action_1})[player1]
                    reward = env_feedback.reward
                    terminate = env_feedback.terminate

                    if 'msg_success' in env_feedback:
                        msg_succeed.append(env_feedback.msg_success)
                    
                    if terminate:
                        if 'action_success' in env_feedback:
                            succeed.append(env_feedback.action_success)
                            action_succeed.append(env_feedback.action_success)
                        else:
                            succeed.append(False)
                        traj_len.append(t + 1)
                        traj['traj_len'] = t + 1

                    traj['reward'].append(reward)
                    traj['pi'].append(action_prob_1)
                    traj['action_p_idx'].append(prev_action_p_idx)
                    traj['action'].append(action_1)
                    
                    prev_action_p_idx = np.concatenate([all_action_encode[action_1], p1_idx_oh])

                else:
                    env_feedback = calendar.proceed({player2: action_2})[player2]
                    reward = env_feedback.reward
                    terminate = env_feedback.terminate

                    if 'msg_success' in env_feedback:
                        msg_succeed.append(env_feedback.msg_success)

                    if terminate:
                        if 'action_success' in env_feedback:
                            succ = env_feedback.action_success
                            succeed.append(succ)
                            action_succeed.append(succ)
                        else:
                            succ = False
                            succeed.append(False)
                        traj_len.append(t + 1)
                        if not succ:
                            p2_wrong_count += 1
                        traj['traj_len'] = t + 1

                    traj['reward'].append(reward)
                    traj['pi'].append(action_prob_2)
                    traj['action_p_idx'].append(prev_action_p_idx)
                    traj['action'].append(action_2)

                    prev_action_p_idx = np.concatenate([all_action_encode[action_2], p2_idx_oh])

            t += 1
        
        trajs.append(traj)

    max_traj_len = max(traj_len)
    accuracy = np.mean(succeed)
    wrong_count = len(succeed) - np.sum(succeed)
    p2_wrong = p2_wrong_count / wrong_count
    msg_acc = np.mean(msg_succeed)
    action_acc = np.mean(action_succeed)

    return trajs, max_traj_len, accuracy, msg_acc, action_acc, p2_wrong, np.mean(traj_len)




def fit_qnet(trajs, max_traj_len):
    p1_calendars = []
    p2_calendars = []
    action_p_idxs = []
    q_chosen_idxs = []

    for i, traj in enumerate(trajs):
        traj_len = traj['traj_len']
        p1_calendars.append(traj['p1_calendar'])
        p2_calendars.append(traj['p2_calendar'])
        action_p_idxs.append(
            np.array(traj['action_p_idx']) if traj_len == max_traj_len \
            else np.concatenate([
                np.array(traj['action_p_idx']), 
                np.array([null_action_p_idx] * (max_traj_len - traj_len))
            ], axis = 0)
        )
        q_chosen_idxs.append(np.stack([
            np.array([i] * traj_len), 
            np.arange(traj_len, dtype = np.int32), 
            np.array(traj['action'])
        ], axis = 1))
    
    p1_calendar_np = np.stack(p1_calendars, axis = 0)
    p2_calendar_np = np.stack(p2_calendars, axis = 0)
    action_p_idx_np = np.stack(action_p_idxs, axis = 0)
    q_chosen_idx_np = np.concatenate(q_chosen_idxs, axis = 0)

    target_q_value = qnet_target.get_all_q(p1_calendar_np, p2_calendar_np, action_p_idx_np)

    td_lambda_value_l = []

    for i, traj in enumerate(trajs):
        # y(s_t) = r_t + gamma * [lambda * y(s_t+1) + (1 - lambda) * Q(s_t+1)]
        traj_len = traj['traj_len']
        reward = traj['reward']
        action = traj['action']
        td_lambda_values = [None] * traj_len
        for epoch in reversed(range(traj_len)):
            if epoch == traj_len - 1:
                td_lambda_values[epoch] = reward[epoch]
            else:
                td_lambda_value = \
                    reward[epoch] \
                    + \
                    config.gamma * ( \
                        config.td_lambda * td_lambda_values[epoch + 1] \
                        + \
                        (1 - config.td_lambda) * target_q_value[i][epoch + 1][action[epoch + 1]] \
                    )
                td_lambda_values[epoch] = td_lambda_value
        
        td_lambda_value_l.extend(td_lambda_values)

    td_lambda_value_np = np.array(td_lambda_value_l)
    
    loss = qnet_primary.fit_target_q(p1_calendar_np, p2_calendar_np, action_p_idx_np, q_chosen_idx_np, td_lambda_value_np)
    primary_q_value = qnet_primary.get_all_q(p1_calendar_np, p2_calendar_np, action_p_idx_np)

    return loss, primary_q_value




def policy_grad(trajs, max_traj_len, primary_q_value):
    in_xs = []
    log_prob_idxs = []
    advs = []
    for i, traj in enumerate(trajs):
        traj_len = traj['traj_len']
        in_x_1 = np.stack(
            traj['in_x_1'] if traj_len == max_traj_len else \
            traj['in_x_1'] + [null_in_x] * (max_traj_len - traj_len), 
            axis = 0
        )
        in_x_2 = np.stack(
            traj['in_x_2'] if traj_len == max_traj_len else \
            traj['in_x_2'] + [null_in_x] * (max_traj_len - traj_len), 
            axis = 0
        )
        in_xs.append(in_x_1)
        in_xs.append(in_x_2)
        action = traj['action']
        traj_q = primary_q_value[i][:traj_len]
        baseline = np.mean(traj_q * np.array(traj['pi']), axis = 1)
        current_q = traj_q[range(traj_len), action]
        adv = current_q - baseline
        advs.extend(adv)

        fst_act_player = traj['fst_act_player']
        for t in range(traj_len):
            if t % 2 == fst_act_player:
                log_prob_idxs.append([2 * i, t, action[t]])
            else:
                log_prob_idxs.append([2 * i + 1, t, action[t]])
        
    in_x_np = np.stack(in_xs, axis = 0)
    log_prob_idx_np = np.stack(log_prob_idxs, axis = 0)
    adv_np = np.array(advs)

    loss = policy.train(in_x_np, log_prob_idx_np, adv_np)

    return loss



def main():
    if not config.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    sess_config = tf.ConfigProto(allow_soft_placement = True, log_device_placement = False)
    sess_config.gpu_options.allow_growth = True
    sess = tf.Session(config = sess_config)

    # DONT change players name
    global player1, player2
    player1 = 0 # teacher
    player2 = 1 # student

    global calendar
    calendar = Calendar(player1, player2, \
                    config.step_reward, config.fail_reward, config.succeed_reward, \
                    config.num_slot, dataset_path_prefix = config.dataset_path_prefix)
    assert(config.num_candidate == calendar.num_calendars_)
    assert(config.num_action == calendar.num_actions_)
    assert(config.action_encode_dim == calendar.action_tensor_.shape[1])

    with tf.device(config.gpu_name if config.gpu_name else 'cpu:0'):
        global qnet_primary, qnet_target
        qnet_primary = QNet(config, sess, False)
        qnet_target = QNet(config, sess, True)

        global policy
        if not config.change_pair_test:
            policy = Policy(config, sess, config.policy_name_appendix)
        else:
            policy_1 = Policy(config, sess, config.policy_name_appendix)
            policy_2 = Policy(config, sess, config.policy_name_appendix_2)
    
    if config.change_pair_test:
        ''' 
        if cannot load, check the name of gru!!!
        '''
        policy_name_1 = 'Policy/' if config.policy_name_appendix is None else 'Policy_{}/'.format(config.policy_name_appendix)
        policy_name_2 = 'Policy/' if config.policy_name_appendix_2 is None else 'Policy_{}/'.format(config.policy_name_appendix_2)
        
        saver_1 = tf.train.Saver(var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope = policy_name_1))
        saver_2 = tf.train.Saver(var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope = policy_name_2))
        if os.path.exists(config.load_ckpt + '.index'):
            saver_1.restore(sess, config.load_ckpt)
        else:
            print('load_ckpt does not exist', config.load_ckpt)
        if os.path.exists(config.load_ckpt_2 + '.index'):
            saver_2.restore(sess, config.load_ckpt_2)
        else:
            print('load_ckpt_2 does not exist', config.load_ckpt_2)
            
        change_pair_test(config, policy_1, policy_2, calendar)
        exit()
    
    saver = tf.train.Saver(max_to_keep = None)
    
    ckpt_name = os.path.splitext(os.path.basename(config.ckpt_path))[0]

    if os.path.exists(config.load_ckpt + '.index'):
        saver.restore(sess, config.load_ckpt)
    else:
        print('load_ckpt does not exist')
        init = tf.global_variables_initializer()
        sess.run(init)
        config_name = 'config_' + ckpt_name + '.py'
        # if not os.path.exists(config_name):
        copyfile('config.py', config_name)

    if config.max_len_msg_test:
        max_len_msg_test(config, policy, calendar)
        exit()


    # Get all the variables in the Q primary network.
    q_primary_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope = "primary")
    # Get all the variables in the Q target network.
    q_target_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope = "target")

    # to initialize target QNet to primary QNet
    qnet_target.set_target_net_update_op(q_primary_vars, q_target_vars)
    qnet_target.target_net_update()
    
    global null_action, null_action_p_idx, null_in_x
    null_action = np.zeros([config.action_encode_dim])
    null_action_p_idx = np.zeros([config.action_encode_dim + 2])
    null_in_x = np.zeros([config.num_slot + config.action_encode_dim + 2 + 2])
    global p1_idx_oh, p2_idx_oh
    p1_idx_oh = np.array([0, 1])
    p2_idx_oh = np.array([1, 0])
    global all_action_encode
    all_action_encode = calendar.action_tensor_

    qnet_losses = []
    pg_losses = []
    acc_list = []
    msg_acc_list = []
    action_acc_list = []
    p2_wrong_rate_list = []
    traj_len_list = []

    test_acc = []

    
    for episode in range(config.itr):
        if (episode + 1) % config.test_itr == 0:
            config.test = True
            _, _, acc, _, _, _, _ = sample_trajs()
            test_acc.append(acc)
            print('[test acc]: {}'.format(acc))
            saver.save(sess, config.ckpt_path, global_step = episode + 1)
            print('ckpt saved')
            print()
            config.test = False

        trajs, max_traj_len, acc, msg_acc, action_acc, p2_wrong_rate, traj_len = sample_trajs()
        acc_list.append(acc)
        msg_acc_list.append(msg_acc)
        action_acc_list.append(action_acc)
        p2_wrong_rate_list.append(p2_wrong_rate)
        traj_len_list.append(traj_len)

        qnet_loss, primary_q_value = fit_qnet(trajs, max_traj_len)
        qnet_losses.append(qnet_loss)
        
        if episode % config.qnet_target_update_itr == 0 and episode != 0:
            qnet_target.target_net_update()

        pg_loss = policy_grad(trajs, max_traj_len, primary_q_value)
        pg_losses.append(pg_loss)
        
        if episode % config.print_itr == 0:
            print('[t{}]'.format(episode))
            print('accuracy: ', np.mean(acc_list))
            acc_list = []
            print('msg_acc: ', np.mean(msg_acc_list))
            msg_acc_list = []
            print('action_acc: ', np.mean(action_acc_list))
            action_acc_list = []
            print('qnet loss: ', np.mean(qnet_losses))
            qnet_losses = []
            print('pg loss: ', np.mean(pg_losses))
            pg_losses = []
            print('p2_wrong: ', np.mean(p2_wrong_rate_list))
            p2_wrong_rate_list = []
            print('mean_traj_len: ', np.mean(traj_len_list))
            traj_len_list = []
            print()
    
    np.save(ckpt_name + '_test_acc', np.array(test_acc))



if __name__ == '__main__':
    main()
