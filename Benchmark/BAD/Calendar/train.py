import tensorflow as tf
import numpy as np
from collections import Counter
from tqdm import tqdm
import os
import sys
import pdb
from copy import deepcopy
from shutil import copyfile
from policy import Policy
from reward import Reward
from value import Value
if len(sys.argv) == 1:
    from config import config
else:
    exec("from {} import config".format(sys.argv[1]))
from eval import change_pair_test
if os.path.basename(os.getcwd()) == 'Calendar':
    sys.path.append("../../..")
else:
    sys.path.append("./")
from Game.calendar import Calendar


np.set_printoptions(suppress = True, precision = 4)


def sample_sars(calendar, policy_net, reward_net):
    sars = {
        'belief_1': [], 
        'new_belief_1': [], 
        'belief_2': [], 
        'new_belief_2': [], 
        'p1_cid': [], 
        'p2_cid': [], 
        'cf_1': [],
        'cf_2': [],
        'bad_reward': [],
        'terminal': [],
        'sampled_action': []
    }

    succeed = []
    msg_succeed = []
    action_succeed = []
    traj_len = []
    p2_wrong_count = 0

    for _ in range(config.batch_size if not config.test else config.test_batch_size):
        calendar.reset(train = not config.test)
        p1_cid = calendar.player1_cid_ - 1
        p2_cid = calendar.player2_cid_ - 1

        belief_1 = uniform_belief
        belief_2 = uniform_belief

        t = 0
        terminate = False
        fst_act_player = np.random.randint(0, 2)
        while not terminate:
            if t >= config.terminate_len:
                succeed.append(False)
                if np.random.random() < 0.5:
                    p2_wrong_count += 1
                traj_len.append(t)
                break

            elif t % 2 == fst_act_player:
                belief_2_stack = np.stack([belief_2] * config.num_candidate, axis = 0)
                sampled_action = policy_net.sample_action(all_target, belief_2_stack)
                if not config.test:
                    reward = reward_net.get_reward(all_target_idx, p2_cid, sampled_action, player1)
                    bad_reward = np.sum(belief_1 * reward)

                sampled_real_action = sampled_action[p1_cid]
                belief_update_mask = (sampled_action == sampled_real_action)
                new_belief_pre_norm = belief_update_mask * belief_1
                new_belief_pre_norm_sum = np.sum(new_belief_pre_norm)
                assert(new_belief_pre_norm_sum != 0)
                new_belief = new_belief_pre_norm / new_belief_pre_norm_sum

                calendar.reset(player1_cid = p1_cid + 1, player2_cid = p2_cid + 1)
                env_feedback = calendar.proceed({player1: sampled_real_action})[player1]
                terminate = env_feedback.terminate

                if not config.test:
                    sars['belief_1'].append(belief_1)
                    sars['new_belief_1'].append(new_belief)
                    sars['belief_2'].append(belief_2)
                    sars['new_belief_2'].append(belief_2)
                    sars['cf_1'].append(all_target)
                    sars['cf_2'].append(belief_2_stack)
                    sars['sampled_action'].append(sampled_action)
                    sars['p1_cid'].append(p1_cid)
                    sars['p2_cid'].append(p2_cid)
                    sars['bad_reward'].append(bad_reward)
                    sars['terminal'].append(terminate)

                belief_1 = new_belief

                if 'msg_success' in env_feedback:
                    msg_succeed.append(env_feedback.msg_success)

                if terminate:
                    if 'action_success' in env_feedback:
                        succeed.append(env_feedback.action_success)
                        action_succeed.append(env_feedback.action_success)
                    else: # 'action_success' in env_feedback
                        succeed.append(False)
                    traj_len.append(t + 1)

            else:
                belief_1_stack = np.stack([belief_1] * config.num_candidate, axis = 0)
                sampled_action = policy_net.sample_action(belief_1_stack, all_target)
                if not config.test:
                    reward = reward_net.get_reward(p1_cid, all_target_idx, sampled_action, player2)
                    bad_reward = np.sum(belief_2 * reward)

                sampled_real_action = sampled_action[p2_cid]
                belief_update_mask = (sampled_action == sampled_real_action)
                new_belief_pre_norm = belief_update_mask * belief_2
                new_belief_pre_norm_sum = np.sum(new_belief_pre_norm)
                assert(new_belief_pre_norm_sum != 0)
                new_belief = new_belief_pre_norm / new_belief_pre_norm_sum

                calendar.reset(player1_cid = p1_cid + 1, player2_cid = p2_cid + 1)
                env_feedback = calendar.proceed({player2: sampled_real_action})[player2]
                terminate = env_feedback.terminate

                if not config.test:
                    sars['belief_1'].append(belief_1)
                    sars['new_belief_1'].append(belief_1)
                    sars['belief_2'].append(belief_2)
                    sars['new_belief_2'].append(new_belief)
                    sars['cf_1'].append(belief_1_stack)
                    sars['cf_2'].append(all_target)
                    sars['sampled_action'].append(sampled_action)
                    sars['p1_cid'].append(p1_cid)
                    sars['p2_cid'].append(p2_cid)
                    sars['bad_reward'].append(bad_reward)
                    sars['terminal'].append(terminate)

                belief_2 = new_belief

                if 'msg_success' in env_feedback:
                    msg_succeed.append(env_feedback.msg_success)

                if terminate:
                    if 'action_success' in env_feedback:
                        succ = env_feedback.action_success
                        succeed.append(succ)
                        action_succeed.append(succ)
                    else: # 'action_success' in env_feedback
                        succ = False
                        succeed.append(False)
                    traj_len.append(t + 1)
                    if not succ:
                        p2_wrong_count += 1
            
            t += 1
    
    if not config.test:
        for k, v in sars.items():
            if k in ['cf_1', 'cf_2', 'sampled_action']:
                sars[k] = np.concatenate(v, axis = 0)
            else:
                sars[k] = np.array(v)
    else:
        sars = None
        
    accuracy = np.mean(succeed)
    wrong_count = len(succeed) - np.sum(succeed)
    p2_wrong = p2_wrong_count / wrong_count
    msg_acc = np.mean(msg_succeed)
    action_acc = np.mean(action_succeed)
    
    return sars, accuracy, msg_acc, action_acc, p2_wrong, np.mean(traj_len)


def train_value_net(value_net, sars):
    belief_1 = sars['belief_1']
    new_belief_1 = sars['new_belief_1']
    belief_2 = sars['belief_2']
    new_belief_2 = sars['new_belief_2']
    p1_cid = sars['p1_cid']
    p2_cid = sars['p2_cid']
    bad_reward = sars['bad_reward']
    terminal = sars['terminal']

    next_value = value_net.get_value(new_belief_1, new_belief_2, p1_cid, p2_cid)
    target_value = bad_reward + config.gamma * (1 - terminal) * next_value

    loss, td_error = value_net.train_value(belief_1, belief_2, p1_cid, p2_cid, target_value)

    return loss, td_error


def train_policy_net(policy_net, value_net, sars, td_error):
    size = len(sars['belief_1'])
    cf_1 = sars['cf_1']
    cf_2 = sars['cf_2']
    prob_idx = np.stack([
        np.arange(size * config.num_candidate, dtype = np.int32),
        sars['sampled_action']
    ], axis = 1)
    
    loss, epsilon = policy_net.train_policy(cf_1, cf_2, prob_idx, td_error)

    return loss, epsilon


def main():
    global player1, player2
    player1 = 0
    player2 = 1
    calendar = Calendar(
        player1, player2,
        config.step_reward, config.fail_reward, config.succeed_reward, 
        config.num_slot, dataset_path_prefix = config.dataset_path_prefix
    )
    assert(config.num_candidate == calendar.num_calendars_ - 1)
    assert(config.num_action == calendar.num_actions_)

    global all_target_idx
    all_target_idx = np.arange(config.num_candidate, dtype = np.int32)
    global all_target
    all_target = np.eye(config.num_candidate)
    global uniform_belief
    uniform_belief = np.ones(config.num_candidate) / config.num_candidate
    global all_calendar
    all_calendar = calendar.tensor_[1:]
    
    if not config.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    sess_config = tf.ConfigProto(allow_soft_placement = True, log_device_placement = False)
    sess_config.gpu_options.allow_growth = True
    sess = tf.Session(config = sess_config)

    with tf.device(config.gpu_name if config.gpu_name else 'cpu:0'):
        policy_net = Policy(config, sess, config.policy_name_appendix, all_calendar)
        reward_net = Reward(config, calendar)
        value_net = Value(config, sess)

    saver = tf.train.Saver(max_to_keep = None)

    ckpt_name = os.path.splitext(os.path.basename(config.ckpt_path))[0]

    dir = os.path.dirname(os.path.realpath(__file__))

    if config.change_pair_test:
        policy_1 = policy_net
        policy_2 = Policy(config, sess, config.policy_name_appendix_2, all_calendar)
        policy_1_name = 'Policy_{}/'.format(config.policy_name_appendix) if config.policy_name_appendix is not None else 'Policy/'
        policy_2_name = 'Policy_{}/'.format(config.policy_name_appendix_2) if config.policy_name_appendix_2 is not None else 'Policy/'

        policy_1_saver = tf.train.Saver(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope = policy_1_name))
        policy_2_saver = tf.train.Saver(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope = policy_2_name))

        if os.path.exists(os.path.join(dir, config.load_ckpt + '.index')):
            policy_1_saver.restore(sess, os.path.join(dir, config.load_ckpt))
            print("teacher's policy net restored")
        else:
            print("teacher's policy net ckpt does not exist !")
            print(os.path.join(dir, config.load_ckpt + '.index'))
            exit()
        
        if os.path.exists(os.path.join(dir, config.load_ckpt_2 + '.index')):
            policy_2_saver.restore(sess, os.path.join(dir, config.load_ckpt_2))
            print("student's policy net restored")
        else:
            print("student's policy net ckpt does not exist !")
            print(os.path.join(dir, config.load_ckpt_2 + '.index'))
            exit()

        change_pair_test(calendar, policy_1, policy_2)


    else:
        if os.path.exists(os.path.join(dir, config.load_ckpt + '.index')):
            saver.restore(sess, os.path.join(dir, config.load_ckpt))
            print('restored from load_ckpt')
        else:
            print('load_ckpt does not exist')
            init = tf.global_variables_initializer()
            sess.run(init)
            copied_config_path = os.path.join(dir, 'config_' + ckpt_name + '.py')
            # if not os.path.exists(copied_config_path):
            copyfile(os.path.join(dir, 'config.py'), copied_config_path)
        
        value_net_losses = []
        policy_net_losses = []
        accuracy_l = []
        msg_acc_l = []
        action_acc_l = []
        p2_wrong_l = []
        mean_traj_len_l = []

        test_acc_l = []

        for episode in range(0 if config.ctd_itr is None else config.ctd_itr, config.itr):
            if (episode + 1) % config.test_itr == 0:
                config.test = True
                _, accuracy, msg_acc, action_acc, p2_wrong, mean_traj_len = sample_sars(calendar, policy_net, reward_net)
                test_acc_l.append(accuracy)
                print('[test acc]: {}'.format(accuracy))
                saver.save(sess, config.ckpt_path, global_step = episode + 1)
                print('ckpt saved')
                print()
                config.test = False

            sars, accuracy, msg_acc, action_acc, p2_wrong, mean_traj_len = sample_sars(calendar, policy_net, reward_net)

            accuracy_l.append(accuracy)
            msg_acc_l.append(msg_acc)
            action_acc_l.append(action_acc)
            p2_wrong_l.append(p2_wrong)
            mean_traj_len_l.append(mean_traj_len)

            value_net_loss, td_error = train_value_net(value_net, sars)
            value_net_losses.append(value_net_loss)

            policy_net_loss, epsilon = train_policy_net(policy_net, value_net, sars, td_error)
            policy_net_losses.append(policy_net_loss)

            if episode % config.print_itr == 0:
                print('[t{}]'.format(episode))
                print('accuracy: ', np.mean(accuracy_l))
                accuracy_l = []
                print('msg_acc: ', np.mean(msg_acc_l))
                msg_acc_l = []
                print('action_acc: ', np.mean(action_acc_l))
                action_acc_l = []
                print('value net loss: ', np.mean(value_net_losses))
                value_net_losses = []
                print('policy net loss: ', np.mean(policy_net_losses))
                policy_net_losses = []
                print('p2_wrong: ', np.mean(p2_wrong_l))
                p2_wrong_l = []
                print('mean_traj_len: ', np.mean(mean_traj_len_l))
                mean_traj_len_l = []
                print('epsilon: ', epsilon)
                print()

        np.save(ckpt_name + '_test_acc', np.array(test_acc_l))


if __name__ == '__main__':
    main()
