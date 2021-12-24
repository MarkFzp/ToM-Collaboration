import tensorflow as tf
import numpy as np
import os
import sys
from copy import deepcopy
from shutil import copyfile
from qnet import QNet
from qnet_const_mess import QNet_const_mess
if os.path.basename(os.getcwd()) == 'DIAL':
    sys.path.append("../..")
else:
    sys.path.append("./")
from Game.kitchen import Kitchen
if len(sys.argv) == 1:
    from config import config
else:
    exec("from {} import config".format(sys.argv[1]))

np.set_printoptions(suppress = True, precision = 4)

def run_qnet():
    menus = []
    menu_embeds = []
    p1_targets = []
    for _ in range(config.batch_size):
        kitchen.reset(train = not (config.test or config.change_pair_test or config.constant_message_test))
        menu = kitchen.state_.menu
        menu_embed = kitchen.state_.menu_embed
        p1_target = kitchen.state_.order
        menus.append(menu)
        menu_embeds.append(menu_embed)
        p1_targets.append(p1_target)
    
    menu_embed = np.stack(menu_embeds, axis = 0)
    p1_target = np.array(p1_targets)

    q_idx = []
    q_target_idx_i = []
    q_target_idx_j = []
    terminal = []
    traj_len = []
    success_list = []
    step_reward = []
    p2_wrong_count = 0

    action_tensor = qnet_primary.get_action(menu_embed, p1_target)
    for i, action_seq in enumerate(action_tensor):
        kitchen.reset(menu = menus[i], order = p1_targets[i])

        for ts, action in enumerate(action_seq):
            if ts % 2 == 0:
                player_idx = player1
            else:
                player_idx = player2
            env_feedback = kitchen.proceed({player_idx: action})[player_idx]
            reward = env_feedback.reward
            terminate = env_feedback.terminate
            q_idx.append([i, ts, action])
            q_target_idx_i.append(i)
            q_target_idx_j.append(ts)
            step_reward.append(reward)
            
            if terminate:
                success = env_feedback.success
                success_list.append(success)
                if not success and ts % 2 == 1:
                    p2_wrong_count += 1
                terminal.append(True)
                traj_len.append(ts + 1)
                break
            else:
                terminal.append(False)
    
    q_idx = np.array(q_idx)
    terminal = np.array(terminal)
    step_reward = np.array(step_reward)
    acc = np.mean(success_list)
    mean_traj_len = np.mean(traj_len)
    p2_wrong = p2_wrong_count / (np.sum(np.logical_not(success_list)) + 1e-12)
    
    if not (config.test or config.change_pair_test or config.constant_message_test):
        q_target = np.concatenate([
            qnet_target.get_target_q(menu_embed, p1_target, action_tensor)[:, :-1],
            np.full([config.batch_size, 1], np.nan)
        ], axis = 1)
        q_target_selected = q_target[q_target_idx_i, q_target_idx_j]
        q_spvs = step_reward + config.gamma * np.where(terminal, 0.0, q_target_selected)
        assert(np.nan not in q_spvs)

        q_loss, eps = qnet_primary.train(menu_embed, p1_target, action_tensor, q_spvs, q_idx)
    else:
        q_loss = np.nan
        eps = np.nan

    return acc, mean_traj_len, p2_wrong, q_loss, eps


def main():
    if not config.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    sess_config = tf.ConfigProto(allow_soft_placement = True, log_device_placement = False)
    sess_config.gpu_options.allow_growth = True
    sess = tf.Session(config = sess_config)

    # DONT change players name
    global player1, player2, kitchen
    player1 = 0 # teacher
    player2 = 1 # student
    kitchen = Kitchen(
        player1, player2, 
        config.step_reward, config.fail_reward, config.succeed_reward, 
        range(config.num_attribute), config.num_candidate, config.max_attribute
    )
    
    if config.fixed_menu:
        global fixed_menu
        kitchen.reset()
        fixed_menu = kitchen.state_.menu
        print('use fixed menu ...')
        print(fixed_menu)

    global qnet_primary, qnet_target
    qnet_primary = QNet(config, sess, config.qnet_name_appendix, False)
    qnet_target = QNet(config, sess, config.qnet_name_appendix, True)
    # qnet_const_mess = QNet_const_mess(config, sess, config.qnet_name_appendix)

    saver = tf.train.Saver(max_to_keep = None)
    ckpt_name = os.path.splitext(os.path.basename(config.ckpt_path))[0]

    if os.path.exists(config.ckpt_path + '.index'):
        saver.restore(sess, config.ckpt_path)
    else:
        print('ckpt_path does not exist')
        init = tf.global_variables_initializer()
        sess.run(init)
        # config_name = 'config_' + ckpt_name + '.py'
        # if not os.path.exists(config_name):
        #     copyfile('config.py', config_name)

    if config.qnet_name_appendix is None:
        primary_scope_name = 'QNet_primary/'
        target_scope_name = 'QNet_target/'
        # const_mess_scope_name = 'QNet_const_mess/'
    else:
        primary_scope_name = 'QNet_primary_{}/'.format(config.qnet_name_appendix)
        target_scope_name = 'QNet_target_{}/'.format(config.qnet_name_appendix)
        # const_mess_scope_name = 'QNet_const_mess_{}/'.format(config.qnet_name_appendix)


    # Get all the variables in the Q primary network.
    q_primary_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope = primary_scope_name)
    # Get all the variables in the Q target network.
    q_target_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope = target_scope_name)

    # q_const_mess_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope = const_mess_scope_name)

    # to initialize target QNet to primary QNet
    qnet_target.set_target_net_update_op(q_primary_vars, q_target_vars)
    # qnet_target.target_net_update()


    traj_len_list = []
    acc_list = []
    p2_wrong_list = []
    q_loss_list = []

    test_acc = []
    const_mess_acc = []

    for itr in range(config.itr):
        acc, traj_len, p2_wrong, q_loss, eps = run_qnet()
        acc_list.append(acc)
        traj_len_list.append(traj_len)
        p2_wrong_list.append(p2_wrong)
        q_loss_list.append(q_loss)

        if (itr + 1) % config.test_itr == 0:
            config.test = True
            inner_acc = []
            for _ in range(config.test_batch_size // config.batch_size):
                acc, _, _, _, _ = run_qnet()
                inner_acc.append(acc)
            batch_test_acc = np.mean(inner_acc)
            test_acc.append(batch_test_acc)
            config.test = False

            config.constant_message_test = True
            inner_acc = []
            for _ in range(config.test_batch_size // config.batch_size):
                acc, _, _, _, _ = run_qnet()
                inner_acc.append(acc)
            batch_const_mess_acc = np.mean(inner_acc)
            const_mess_acc.append(batch_const_mess_acc)
            config.constant_message_test = False

            print('[test acc]: {}'.format(batch_test_acc))
            print('[const mess acc]: {}'.format(batch_const_mess_acc))
            saver.save(sess, config.ckpt_path, global_step = itr + 1)
            print('ckpt saved')
            print()


        if itr % config.qnet_target_update_itr == 0:
            qnet_target.target_net_update()

        if itr % config.print_itr == 0:
            print('[t{}]'.format(itr))
            print('accuracy: ', np.mean(acc_list))
            acc_list = []
            print('qnet loss: ', np.mean(q_loss_list))
            q_loss_list = []
            print('p2_wrong: ', np.mean(p2_wrong_list))
            p2_wrong_list = []
            print('mean_traj_len: ', np.mean(traj_len_list))
            traj_len_list = []
            print('epsilon: ', eps)
            print()
        
    np.save(ckpt_name + '_test_acc', np.array(test_acc))
    np.save(ckpt_name + '_const_mess_acc', np.array(const_mess_acc))



if __name__ == '__main__':
    main()
