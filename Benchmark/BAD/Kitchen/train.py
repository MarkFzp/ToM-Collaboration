import tensorflow as tf
import numpy as np
from collections import Counter
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
from eval import change_pair_test, test, order_free_test
if os.path.basename(os.getcwd()) == 'Kitchen':
    sys.path.append("../../..")
else:
    sys.path.append("./")
from Game.kitchen import Kitchen


np.set_printoptions(suppress = True, precision = 4)


def train_reward_net(kitchen, reward_net, reward_saver, sess):
    reward_loss_l = []
    reward_stats_count = [[0, 0] for _ in range(config.max_attribute)]
    reward_stats_l1 = [[[], []] for _ in range(config.max_attribute)]
    total_count = 0
    large_dev_count = 0
    for itr in range(config.reward_itr):
        menu_l = []
        workplace_l = []
        target_l = []
        action_l = []
        reward_l = []

        reward_stats_idx = [[[], []] for _ in range(config.max_attribute)]
        batch_idx = 0

        for _ in range(config.batch_size):
            kitchen.reset()
            start_ret = kitchen.start()
            p1_start_ret = start_ret[player1]
            menu = p1_start_ret.observation.menu_embed
            workplace = deepcopy(p1_start_ret.observation.workplace_embed)
            p1_target = p1_start_ret.private
            p1_target_ingred = menu[np.where(p1_target)[0][0]]

            t = 0
            terminate = False
            while not terminate:
                menu_l.append(menu)
                workplace_l.append(workplace)
                target_l.append(p1_target)
                if np.random.random() < config.eps_greedy[t]:
                    action = np.random.choice(np.where(p1_target_ingred - workplace)[0])
                else:
                    action = np.random.randint(config.num_action)
                action_l.append(action)

                if t % 2 == 0:
                    env_feedback = kitchen.proceed({player1: action})[player1]
                else:
                    env_feedback = kitchen.proceed({player2: action})[player2]
                terminate = env_feedback.terminate
                reward = env_feedback.reward
                workplace = deepcopy(env_feedback.observation.workplace_embed)
                reward_l.append(reward)

                if reward == config.step_reward:
                    reward_stats_count[t][0] += 1
                    reward_stats_idx[t][0].append(batch_idx)
                elif reward == config.fail_reward:
                    reward_stats_count[t][1] += 1
                    reward_stats_idx[t][1].append(batch_idx)
                elif reward == config.succeed_reward:
                    reward_stats_count[-1][0] += 1
                    reward_stats_idx[-1][0].append(batch_idx)
                else:
                    raise Exception()
                
                total_count += 1
                batch_idx += 1
                t += 1
        
        
        menu_np = np.array(menu_l)
        workplace_np = np.array(workplace_l)
        target_np = np.array(target_l)
        action_np = np.array(action_l)
        reward_np = np.array(reward_l)

        reward_loss, l1_diff, reward_pred = reward_net.train_reward(menu_np, workplace_np, target_np, action_np, reward_np)
        reward_loss_l.append(reward_loss)

        if config.reward_architecture == 'adhoc':
            assert(np.sum(l1_diff) == 0)

        large_dev_count += len(np.where(l1_diff > config.large_dev_thres)[0])

        for i in range(config.max_attribute):
            reward_stats_l1[i][0].extend(l1_diff[reward_stats_idx[i][0]])
            reward_stats_l1[i][1].extend(l1_diff[reward_stats_idx[i][1]])

        if itr % config.print_itr == 0:
            print('[t{}]'.format(itr))
            print('reward loss: ', np.mean(reward_loss_l))
            print('large deviation: {:.5%}'.format(large_dev_count / total_count))
            print('succ-{:.2%}'.format(reward_stats_count[-1][0]/total_count))
            for i, pair in enumerate(reward_stats_count, start = 1):
                step_count, fail_count = pair
                if i != len(reward_stats_count):
                    print('{}: step-{:.2%} fail-{:.2%}'.format(i, step_count/total_count, fail_count/total_count))
                else:
                    print('{}: fail-{:.2%}'.format(i, fail_count/total_count))
            print('succ-{:.4f}'.format(np.mean(reward_stats_l1[-1][0])))
            for i, pair in enumerate(reward_stats_l1, start = 1):                
                step_l1, fail_l1 = pair
                if i != len(reward_stats_l1):
                    if len(step_l1) > 0 and len(fail_l1) > 0:
                        print('{}: step-{:.4f} fail-{:.4f}'.format(i, np.mean(step_l1), np.mean(fail_l1)))
                else:
                    if len(fail_l1) > 0:
                        print('{}: fail-{:.4f}'.format(i, np.mean(fail_l1)))
            print()

            reward_loss_l = []
            reward_stats_count = [[0, 0] for _ in range(config.max_attribute)]
            reward_stats_l1 = [[[], []] for _ in range(config.max_attribute)]
            total_count = 0
            large_dev_count = 0
            
            if config.debug:
                print('real r : prediced r')
                print(np.stack([reward_np, reward_pred], axis = -1))
                input('press ENTER to proceed...')
        
        if (itr + 1) % config.save_itr == 0 and itr != 0 and config.reward_architecture != 'adhoc':
            reward_saver.save(sess, config.reward_ckpt_path)
            print('reward net ckpt saved')
            print()



def sample_sars(kitchen, policy_net, reward_net):
    tea_sars = {
        'menu': [], 
        'menu_tile': [], 
        'workplace': [], 
        'workplace_tile': [],
        'new_workplace': [], 
        'belief': [], 
        # 'belief_tile': [], 
        'new_belief': [], 
        'sampled_action': [], 
        'prob_pre_anneal': [], 
        'p1_target': [], 
        'bad_reward': [], 
        'terminal': []
    }
    stu_sars = {
        'empty': True, 
        'menu': [], 
        'workplace': [], 
        'new_workplace': [], 
        'belief': [], 
        'new_belief': [], 
        'sampled_action': [], 
        'prob_pre_anneal': [], 
        'p1_target': [], 
        'bad_reward': [], 
        'terminal': []
    }

    succeed = []
    traj_len = []
    p2_wrong_count = 0
    # updated_belief_len = Counter()

    for _ in range(config.batch_size if not config.test else config.test_batch_size):
        if config.fixed_menu:
            kitchen.reset(menu = fixed_menu)
        else:
            kitchen.reset(train = not config.test)
        start_ret = kitchen.start()
        p1_start_ret = start_ret[player1]

        menu = p1_start_ret.observation.menu_embed
        menu_tile = np.stack([menu] * config.num_candidate, axis = 0)
        menu_3d = np.expand_dims(menu, axis = 0)

        workplace = deepcopy(p1_start_ret.observation.workplace_embed)

        p1_target = p1_start_ret.private
        # p1_target_tile = np.stack([p1_target] * config.num_candidate, axis = 0)
        p1_target_2d = np.expand_dims(p1_target, axis = 0)
        p1_target_idx = np.where(p1_target)[0][0]

        belief = uniform_belief

        t = 0
        terminate = False
        while not terminate:
            if config.train_one_step and t > 0:
                succeed.append(True)
                traj_len.append(t + 1)
                break

            if t % 2 == 0:
                workplace_tile = np.stack([workplace] * config.num_candidate, axis = 0)
                # belief_tile = np.stack([belief] * config.num_candidate, axis = 0)
                sampled_action, prob_pre_anneal = policy_net.sample_action(menu_tile, workplace_tile, all_target)
                reward = reward_net.get_reward(menu_tile, workplace_tile, all_target, sampled_action)
                bad_reward = np.sum(belief * reward)

                sampled_real_action = sampled_action[p1_target_idx]
                belief_update_mask = (sampled_action == sampled_real_action)
                new_belief_pre_norm = belief_update_mask * belief
                new_belief_pre_norm_sum = np.sum(new_belief_pre_norm)
                assert(new_belief_pre_norm_sum != 0)
                new_belief = new_belief_pre_norm / new_belief_pre_norm_sum

                # if t == 0:
                #     updated_belief_len[len(np.where(new_belief)[0])] += 1

                env_feedback = kitchen.proceed({player1: sampled_real_action})[player1]
                terminate = env_feedback.terminate
                new_workplace = deepcopy(env_feedback.observation.workplace_embed)

                tea_sars['menu'].append(menu)
                tea_sars['menu_tile'].append(menu_tile)
                tea_sars['workplace'].append(workplace)
                tea_sars['workplace_tile'].append(workplace_tile)
                tea_sars['new_workplace'].append(new_workplace)
                tea_sars['belief'].append(belief)
                # tea_sars['belief_tile'].append(belief_tile)
                tea_sars['new_belief'].append(new_belief)
                tea_sars['sampled_action'].append(sampled_action)
                tea_sars['prob_pre_anneal'].append(prob_pre_anneal)
                tea_sars['p1_target'].append(p1_target)
                tea_sars['bad_reward'].append(bad_reward)
                tea_sars['terminal'].append(terminate)

                belief = new_belief
                workplace = new_workplace

                if terminate:
                    succeed.append(env_feedback.success)
                    traj_len.append(t + 1)

            else:
                workplace_2d = np.expand_dims(workplace, axis = 0)
                workplace_tile = np.stack([workplace] * config.num_candidate, axis = 0)
                sampled_action, prob_pre_anneal = policy_net.sample_action(
                    menu_3d, 
                    workplace_2d, 
                    np.expand_dims(belief, axis = 0)
                )
                if not config.strict_imp:
                    reward = reward_net.get_reward(menu_tile, workplace_tile, all_target, np.concatenate([sampled_action] * config.num_candidate, axis = 0))
                    bad_reward = np.sum(belief * reward)
                else:
                    reward = reward_net.get_reward(menu_3d, workplace_2d, p1_target_2d, sampled_action)[0]
                    bad_reward = reward

                sampled_action = sampled_action[0]
                prob_pre_anneal = prob_pre_anneal[0]
                
                env_feedback = kitchen.proceed({player2: sampled_action})[player2]
                terminate = env_feedback.terminate
                new_workplace = deepcopy(env_feedback.observation.workplace_embed)
                
                stu_sars['empty'] = False
                stu_sars['menu'].append(menu)
                stu_sars['workplace'].append(workplace)
                stu_sars['new_workplace'].append(new_workplace)
                stu_sars['belief'].append(belief)
                stu_sars['new_belief'].append(belief)
                stu_sars['sampled_action'].append(sampled_action)
                stu_sars['prob_pre_anneal'].append(prob_pre_anneal)
                stu_sars['p1_target'].append(p1_target)
                stu_sars['bad_reward'].append(bad_reward)
                stu_sars['terminal'].append(terminate)

                workplace = new_workplace

                if terminate:
                    succ = env_feedback.success
                    succeed.append(succ)
                    traj_len.append(t + 1)
                    if not succ:
                        p2_wrong_count += 1
            
            t += 1
    
    for k, v in tea_sars.items():
        if k in ['menu_tile', 'workplace_tile', 'belief_tile']:
            tea_sars[k] = np.concatenate(v, axis = 0)
        else:
            tea_sars[k] = np.array(v)
    
    if not stu_sars['empty']:
        for k, v in stu_sars.items():
            stu_sars[k] = np.array(v)
    
    accuracy = np.mean(succeed)
    wrong_count = len(succeed) - np.sum(succeed)
    p2_wrong = p2_wrong_count / wrong_count

    # print(updated_belief_len)
    
    return tea_sars, stu_sars, accuracy, p2_wrong, np.mean(traj_len)


def train_value_net(value_net, tea_sars, stu_sars):
    if not stu_sars['empty']:
        menu_np = np.concatenate([tea_sars['menu'], stu_sars['menu']], axis = 0)
        workplace_np = np.concatenate([tea_sars['workplace'], stu_sars['workplace']], axis = 0)
        new_workplace_np = np.concatenate([tea_sars['new_workplace'], stu_sars['new_workplace']], axis = 0)
        belief_np = np.concatenate([tea_sars['belief'], stu_sars['belief']], axis = 0)
        new_belief_np = np.concatenate([tea_sars['new_belief'], stu_sars['new_belief']], axis = 0)
        p1_target_np = np.concatenate([tea_sars['p1_target'], stu_sars['p1_target']], axis = 0)
        bad_reward_np = np.concatenate([tea_sars['bad_reward'], stu_sars['bad_reward']], axis = 0)
        terminal_mask_np = np.concatenate([tea_sars['terminal'], stu_sars['terminal']], axis = 0)
    else:
        menu_np = tea_sars['menu']
        workplace_np = tea_sars['workplace']
        new_workplace_np = tea_sars['new_workplace']
        belief_np = tea_sars['belief']
        new_belief_np = tea_sars['new_belief']
        p1_target_np = tea_sars['p1_target']
        bad_reward_np = tea_sars['bad_reward']
        terminal_mask_np = tea_sars['terminal']

    next_value = value_net.get_value(menu_np, new_workplace_np, new_belief_np, p1_target_np)
    target_value = bad_reward_np + config.gamma * (1 - terminal_mask_np) * next_value

    if not config.train_one_step:
        loss, td_error = value_net.train_value(menu_np, workplace_np, belief_np, p1_target_np, target_value)
    else:
        loss = np.nan
        td_error = bad_reward_np

    return loss, td_error


def train_policy_net(policy_net, value_net, tea_sars, stu_sars, td_error):
    tea_size = len(tea_sars['menu'])
    stu_size = len(stu_sars['menu'])

    stu_idx_fed = not stu_sars['empty']

    if stu_idx_fed:
        menu_np = np.concatenate([tea_sars['menu_tile'], stu_sars['menu']], axis = 0)
        workplace_np = np.concatenate([tea_sars['workplace_tile'], stu_sars['workplace']], axis = 0)
        # belief_np = np.concatenate([tea_sars['belief_tile'], stu_sars['belief']], axis = 0)
        # target_np = np.concatenate([
        #     np.concatenate([all_target] * tea_size, axis = 0), 
        #     np.stack([p2_target] * stu_size, axis = 0)
        # ], axis = 0)
        cf_np = np.concatenate([np.concatenate([all_target] * tea_size, axis = 0), stu_sars['belief']], axis = 0)
        tea_prob_idx = np.stack([
            np.arange(tea_size * config.num_candidate, dtype = np.int32), 
            tea_sars['sampled_action'].reshape([-1])
        ], axis = -1)
        stu_prob_idx = np.stack([
            np.arange(tea_size * config.num_candidate, tea_size * config.num_candidate + stu_size, dtype = np.int32),
            stu_sars['sampled_action']
        ], axis = -1)
    
    else:
        menu_np = tea_sars['menu_tile']
        workplace_np = tea_sars['workplace_tile']
        # belief_np = tea_sars['belief_tile']
        # target_np = np.concatenate([all_target] * tea_size, axis = 0)
        cf_np = np.concatenate([all_target] * tea_size, axis = 0)
        tea_prob_idx = np.stack([
            np.arange(tea_size * config.num_candidate, dtype = np.int32), 
            tea_sars['sampled_action'].reshape([-1])
        ], axis = -1)
        stu_prob_idx = None
    
    loss, epsilon = policy_net.train_policy(menu_np, workplace_np, cf_np, tea_prob_idx, stu_prob_idx, stu_idx_fed, td_error)

    return loss, epsilon





def main():
    global p2_target
    p2_target = np.zeros([config.num_candidate])
    global p2_target_2d
    p2_target_2d = np.expand_dims(p2_target, axis = 0)
    global all_target
    all_target = np.eye(config.num_candidate)
    global uniform_belief
    uniform_belief = np.ones(config.num_candidate) / config.num_candidate
    global uniform_belief_tile
    uniform_belief_tile = np.stack([uniform_belief] * config.num_candidate, axis = 0)
    
    if not config.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    sess_config = tf.ConfigProto(allow_soft_placement = True, log_device_placement = False)
    sess_config.gpu_options.allow_growth = True
    sess = tf.Session(config = sess_config)

    # DONT change players name
    global player1, player2
    player1 = 0 # teacher
    player2 = 1 # student
    kitchen = Kitchen(
        player1, player2, 
        config.step_reward, config.fail_reward, config.succeed_reward, 
        range(config.num_attribute), config.num_candidate, config.max_attribute, 
        config.dataset_path
    )
    if config.fixed_menu:
        global fixed_menu
        kitchen.reset()
        fixed_menu = kitchen.state_.menu
        print('use fixed menu ...')
        print(fixed_menu)

    policy_net = Policy(config, sess, config.policy_name_appendix)
    reward_net = Reward(config, sess)
    value_net = Value(config, sess)

    saver = tf.train.Saver(max_to_keep = None)
    if config.reward_architecture == 'adhoc':
        reward_saver = None
    else:
        reward_saver = tf.train.Saver(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope = 'Reward/'))

    dir = os.path.dirname(os.path.realpath(__file__))
    ckpt_name = os.path.splitext(os.path.basename(config.ckpt_path))[0]

    if config.change_pair_test:
        if config.reward_architecture == 'adhoc':
            print('using adhoc reward net !!')
        elif os.path.exists(os.path.join(dir, config.reward_ckpt_path + '.index')):
            reward_saver.restore(sess, os.path.join(dir, config.reward_ckpt_path))
            print('restored reward net')
        else:
            print('reward net ckpt does not exist !')
            exit()

        policy_tea = policy_net
        policy_stu = Policy(config, sess, config.policy_name_appendix_2)
        policy_tea_name = 'Policy_{}/'.format(config.policy_name_appendix) if config.policy_name_appendix is not None else 'Policy/'
        policy_stu_name = 'Policy_{}/'.format(config.policy_name_appendix_2) if config.policy_name_appendix_2 is not None else 'Policy/'

        policy_tea_saver = tf.train.Saver(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope = policy_tea_name))
        policy_stu_saver = tf.train.Saver(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope = policy_stu_name))
        # print(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope = policy_tea_name))
        # policy_stu_saver = tf.train.Saver(
        #     dict([(var.name.replace(policy_stu_name, 'Policy'), var) 
        #     for var in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope = policy_stu_name)])
        # )
        # print(dict([(var.name.replace(policy_stu_name, 'Policy'), var) 
        #     for var in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope = policy_stu_name)]))
        # input()

        if os.path.exists(os.path.join(dir, config.load_ckpt + '.index')):
            policy_tea_saver.restore(sess, os.path.join(dir, config.load_ckpt))
            print("teacher's policy net restored")
        else:
            print("teacher's policy net ckpt does not exist !")
            print(os.path.join(dir, config.load_ckpt + '.index'))
            exit()
        
        if os.path.exists(os.path.join(dir, config.load_ckpt_2 + '.index')):
            policy_stu_saver.restore(sess, os.path.join(dir, config.load_ckpt_2))
            print("student's policy net restored")
        else:
            print("student's policy net ckpt does not exist !")
            print(os.path.join(dir, config.load_ckpt_2 + '.index'))
            exit()

        change_pair_test(kitchen, policy_tea, policy_stu, reward_net)


    else:
        if config.reward_itr == 0:
            if config.load_ckpt is not None and os.path.exists(os.path.join(dir, config.load_ckpt + '.index')):
                saver.restore(sess, os.path.join(dir, config.load_ckpt))
                print('restored from load_ckpt')
                if config.reward_architecture == 'adhoc':
                    print('using adhoc reward net !!')
            else:
                print('ckpt_path does not exist / not loading ckpt')
                init = tf.global_variables_initializer()
                sess.run(init)
                if config.reward_architecture == 'adhoc':
                    print('using adhoc reward net !!')
                elif os.path.exists(os.path.join(dir, config.reward_ckpt_path + '.index')):
                    reward_saver.restore(sess, os.path.join(dir, config.reward_ckpt_path))
                    print('restored reward net')
                else:
                    print('reward net ckpt does not exist !')
                # copied_config_path = os.path.join(dir, 'config_' + ckpt_name + '.py')
                # # if not os.path.exists(copied_config_path):
                # copyfile(os.path.join(dir, 'config.py'), copied_config_path)
        else:
            if config.reward_architecture == 'adhoc':
                print('using adhoc reward net !!, not need to train')
                exit()
            elif os.path.exists(os.path.join(dir, config.reward_ckpt_path + '.index')):
                reward_saver.restore(sess, os.path.join(dir, config.reward_ckpt_path))
                print('restored reward net')
            else:
                init = tf.global_variables_initializer()
                sess.run(init)
                print('reward net ckpt does not exist !')
            
            train_reward_net(kitchen, reward_net, reward_saver, sess)
            exit()
        
        if config.test:
            test(kitchen, policy_net, reward_net)
        
        elif config.order_free_test:
            order_free_test(kitchen, policy_net, reward_net)
        
        else:
            value_net_losses = []
            policy_net_losses = []
            accuracy_l = []
            p2_wrong_l = []
            mean_traj_len_l = []

            test_acc = []

            for episode in range(0 if config.continue_itr is None else config.ctd_itr, config.itr):
                if (episode + 1) % config.test_itr == 0:
                    config.test = True
                    _, _, accuracy, _, _ = sample_sars(kitchen, policy_net, reward_net)
                    test_acc.append(accuracy)
                    print('[test acc]: {}'.format(accuracy))
                    saver.save(sess, config.ckpt_path, global_step = episode + 1)
                    print('ckpt saved')
                    print()
                    config.test = False

                tea_sars, stu_sars, accuracy, p2_wrong, mean_traj_len = sample_sars(kitchen, policy_net, reward_net)
                accuracy_l.append(accuracy)
                p2_wrong_l.append(p2_wrong)
                mean_traj_len_l.append(mean_traj_len)

                value_net_loss, td_error = train_value_net(value_net, tea_sars, stu_sars)
                value_net_losses.append(value_net_loss)

                policy_net_loss, epsilon = train_policy_net(policy_net, value_net, tea_sars, stu_sars, td_error)
                policy_net_losses.append(policy_net_loss)

                if episode % config.print_itr == 0:
                    print('[t{}]'.format(episode))
                    print('accuracy: ', np.mean(accuracy_l))
                    accuracy_l = []
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

                # if (episode + 1) % config.save_itr == 0 and episode != 0:
                #     saver.save(sess, config.ckpt_path)
                #     print('ckpt saved')
                #     print()

            if config.continue_itr != 0:
                np.save(ckpt_name + '_ct{}_test_acc'.format(config.continue_itr), np.array(test_acc))
            else:
                np.save(ckpt_name + '_test_acc', np.array(test_acc))


if __name__ == '__main__':
    main()
