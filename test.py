import sys
import os
from easydict import EasyDict as edict
from tqdm import tqdm
from collections import defaultdict
import numpy as np
import tensorflow as tf
from pprint import pprint as brint

from interact import play, train_step
from Agent.agent import Agent
from Game.kitchen import Kitchen

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
    exp_folder = sys.argv[1]

    if not os.path.isdir(os.path.join('./Experiments', exp_folder)):
        print('Cannot find target folder')
        exit()
    if not os.path.exists(os.path.join('./Experiments', exp_folder, 'config.py')):
        print('Cannot find config.py in target folder')
        exit()

    exec('from Experiments.%s.config import configs' % exp_folder, globals())

    tfconfig = tf.ConfigProto(allow_soft_placement = True, log_device_placement = False)
    tfconfig.gpu_options.allow_growth = True
    sess = tf.Session(config = tfconfig)
    agentA = Agent(sess, configs, 'A', True, False, False)
    agentB = Agent(sess, configs, 'B', False, True, False)
    agents = [agentA, agentB]
    init = tf.global_variables_initializer()
    sess.run(init)
    ckpt_dir = os.path.join('Experiments', exp_folder, configs[0].ckpt_dir)
    agentA.restore_ckpt(ckpt_dir)
    agentB.restore_ckpt(ckpt_dir)

    game = Kitchen(agentA.name_, agentB.name_, configs[1].step_reward,
                   configs[1].fail_reward, configs[1].success_reward,
                   np.arange(configs[1].num_actions), configs[1].num_dishes,
                   configs[1].max_ingredients, os.path.join('/home/luyao/Datasets/Kitchen',
                   'dataset%d_with_10_ingred_4_dish_5_maxingred_1000000_size_0.7_ratio.npz' % configs[0].split))


    np.random.seed(425)
    test_iters = int(1e4)
    records = []
    total_rewards = []
    errors = np.zeros(2)
    belief_correct = 0
    unique_id = [0, 0]
    have_unique_id = [0, 0]
    bad_A_error = 0
    rewards_UI_nonUI = [0, 0]
    use_UI_nonUI = [0, 0]
    ui_qs = []
    non_ui_qs = []
    ui_qs_max = []
    non_ui_qs_max = []
    for tit in tqdm(range(1, test_iters + 1)):
        epsilons = [0, 0]
        sar_sequence = play(agentA, agentB, game, epsilons, False)
        total_reward = np.sum([step[2] for step in sar_sequence[1:]])
        total_rewards.append(total_reward)
        success = (sar_sequence[-1][2] == (configs[1].success_reward + configs[1].step_reward))
        records.append(success)
        agentB_belief_correct = (np.max(sar_sequence[-1][3]) == np.sum(sar_sequence[-1][3] * sar_sequence[0][3]))
        belief_correct += agentB_belief_correct
        menu = sar_sequence[0][2]
        unique_ids = (np.sum(menu > 0, 0) == 1) * (menu[int(np.nonzero(sar_sequence[0][3])[0]), :] > 0)
        non_unique_ids = 1.0 * (menu[int(np.nonzero(sar_sequence[0][3])[0]), :] > 0) - unique_ids
        unique_id[1 - success] += np.sum(unique_ids * np.squeeze(sar_sequence[1][0]))
        if np.sum(unique_ids):
            rewards_UI_nonUI[int(np.sum(unique_ids * np.squeeze(sar_sequence[1][0])))] += sar_sequence[-1][2]
            use_UI_nonUI[int(np.sum(unique_ids * np.squeeze(sar_sequence[1][0])))] += 1
        have_unique_id[1 - success] += np.sum(unique_ids) > 0
        if np.sum(unique_ids) * np.sum(non_unique_ids) > 0:
            ui_qs.append(np.sum(sar_sequence[1][4] * unique_ids) / np.sum(unique_ids))
            non_ui_qs.append(np.sum(sar_sequence[1][4] * (non_unique_ids)) / np.sum(non_unique_ids))
            ui_qs_max.append(np.max(sar_sequence[1][4] * unique_ids))
            non_ui_qs_max.append(np.max(sar_sequence[1][4] * (non_unique_ids)))
        if not success:# and np.sum(unique_ids) > 0:
            errors[len(sar_sequence) % 2] += 1
            if np.sum(unique_ids) > 0 and np.sum(unique_ids * np.squeeze(sar_sequence[1][0])) == 0 and len(sar_sequence) % 2 == 1:
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
            
    print('average successful rate: %f, average reward: %f' % (np.mean(records), np.mean(total_rewards)))
    print('agents error rates', errors / np.sum(errors))
    print('error because of non-ui', bad_A_error / np.sum(errors))
    print('agent B belief correct ratio: %f' % (1.0 * belief_correct / test_iters))
    print('agent A use unique identifier ratio:', np.array(unique_id) / np.array(have_unique_id))
    print('has unique identifier ratio:',
          np.array(have_unique_id) / np.array([test_iters - np.sum(errors), np.sum(errors)]))
    print('rewards not using UI, using UI', np.array(rewards_UI_nonUI) / np.array(use_UI_nonUI))
    print('UI average q: %f, non-UI average q: %f' % (np.mean(ui_qs), np.mean(non_ui_qs)))
    print('Average UI > non-UI ratio: %f' % np.mean(np.array(ui_qs) > np.array(non_ui_qs)))
    print('UI max q: %f, non-UI max q: %f' % (np.mean(ui_qs_max), np.mean(non_ui_qs_max)))
    print('max UI > non-UI ratio: %f' % np.mean(np.array(ui_qs_max) > np.array(non_ui_qs_max)))

    return

if __name__ == '__main__':
    main()
