import sys
import os
import logging
from easydict import EasyDict as edict
from tqdm import tqdm
from collections import defaultdict
import numpy as np
import tensorflow as tf
from pprint import pprint as brint

from train import *
from Agent import Agent
from kitchen import *

import pdb


def sequence_debug(sar_sequence):
    order = int(np.nonzero(sar_sequence[0][3])[0])
    menu_nz = np.nonzero(sar_sequence[0][2])
    menu = defaultdict(list)
    best = []
    for i in range(len(menu_nz[0])):
        for m in range(int(sar_sequence[0][2][menu_nz[0][i], menu_nz[1][i]])):
            menu[menu_nz[0][i]].append(menu_nz[1][i])
    actions = []
    names = 'AB'
    for idx, step in enumerate(sar_sequence[1:]):
        actions.append((names[idx % 2], int(np.nonzero(step[1])[2]), step[3], step[4], step[5],
                        step[6][0, 0, ...]))
    return {'menu': menu, 'actions': actions, 'order': order}


def main():
    exp_folder = config.exp_folder

    if not os.path.isdir(exp_folder):
        print('Cannot find target folder')
        exit()
    if not os.path.exists(os.path.join(exp_folder, 'config.py')):
        print('Cannot find config.py in target folder')
        exit()

    # exec('from config import config' % exp_folder, globals())

    if not config.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    sess_config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    sess_config.gpu_options.allow_growth = True
    sess = tf.Session(config=sess_config)

    student = Agent(config, sess)
    teacher = Agent(config, sess)

    kitchen = Kitchen('teacher', 'student', \
                      config.step_reward, config.fail_reward, config.succeed_reward, \
                      range(config.num_attribute), config.num_candidate, config.max_attribute, config.dataset)

    qnet_primary = QNet(config, sess, False)
    qnet_target = QNet(config, sess, True)
    buffer = ReplayBuffer(config)

    if config.log:
        logger = logging.getLogger()
        logger.handlers = []
        file_handler = logging.FileHandler(os.path.join(config.log_dir, config.log_file))
        console_handler = logging.StreamHandler(sys.stdout)
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        logger.setLevel(logging.INFO)

    saver = tf.train.Saver()

    ckpt_status = tf.train.get_checkpoint_state('./')
    if ckpt_status:
        saver.restore(sess, ckpt_status.model_checkpoint_path)
    else:
        print('ckpt_path does not exist')
        init = tf.global_variables_initializer()
        sess.run(init)

    np.random.seed(425)
    test_iters = int(1e4)
    records = []
    total_rewards = []
    errors = np.zeros(2)
    unique_id = [0, 0]
    have_unique_id = [0, 0]
    bad_A_error = 0
    rewards_UI_nonUI = [0, 0]
    use_UI_nonUI = [0, 0]
    for tit in tqdm(range(test_iters)):
        epsilon = 0
        reward, ts, ss, target, success = buffer_tranjectory(sess, student, teacher, kitchen, buffer, qnet_primary, epsilon, tit)
        total_rewards.append(reward)
        records.append(success)
        menu = np.reshape(ts[0][0][0, 0, :40], [config.num_candidate, config.num_ingredient])
        t_choice = np.zeros(config.num_ingredient)
        t_choice[ts[0][3]] = 1
        unique_ids = (np.sum(menu > 0, 0) == 1) * (target > 0)
        unique_id[1 - success] += np.sum(unique_ids * np.squeeze(t_choice))
        if np.sum(unique_ids) > 0:
            rewards_UI_nonUI[int(np.sum(unique_ids * np.squeeze(t_choice)))] += reward
            use_UI_nonUI[int(np.sum(unique_ids * np.squeeze(t_choice)))] += 1
        have_unique_id[1 - success] += np.sum(unique_ids) > 0
        if not success:  # and np.sum(unique_ids) > 0:
            errors[(len(ts) + len(ss)) % 2] += 1
            if np.sum(unique_ids) > 0 and np.sum(unique_ids * np.squeeze(t_choice)) == 0 and (len(ts) + len(ss)) % 2 == 1:
                bad_A_error += 1
            # if np.sum(unique_ids) > 0 and np.sum(unique_ids * np.squeeze(sar_sequence[1][0])) == 0:
            #     debug_pack = sequence_debug(sar_sequence)
            #     brint(debug_pack)
            #     pdb.set_trace()
            # if len(sar_sequence) % 2 == 0:
            #     debug_pack = sequence_debug(sar_sequence)
            #     brint(debug_pack)
            #     pdb.set_trace()
            # if not agentB_belief_correct:
            # debug_pack = sequence_debug(sar_sequence)
            # brint(debug_pack)
            # pdb.set_trace()

    error_nonUI = bad_A_error / np.sum(errors)
    t_UI = np.array(unique_id) / np.array(have_unique_id)
    has_UI = np.array(have_unique_id) / np.array([test_iters - np.sum(errors), np.sum(errors)])
    r_UI = np.array(rewards_UI_nonUI) / np.array(use_UI_nonUI)

    logging.info('average successful rate: %f, average reward: %f' % (np.mean(records), np.mean(total_rewards)))
    logging.info('teacher error rates: %f, student error rate: %f' % (errors[1] / np.sum(errors), errors[0] / np.sum(errors)))
    logging.info('error because of non-ui: %f' % error_nonUI)
    logging.info('teacher use unique identifier ratio: %s '% t_UI)
    logging.info('has unique identifier ratio: %s' % has_UI)
    logging.info('rewards not using UI, using UI: %s' % r_UI)

    return


if __name__ == '__main__':
    main()
