import sys
import os
from easydict import EasyDict as edict
from tqdm import tqdm
from collections import defaultdict
import numpy as np
import tensorflow as tf
from pprint import pprint as brint

from interactC import play, train_step
from AgentC.agent import Agent
from Game.calendar import Calendar

import pdb

def main():
    exp_folder = sys.argv[1]
    device_idx = '0' if len(sys.argv) == 2 else sys.argv[2]

    if not os.path.isdir(os.path.join('./Experiments', exp_folder)):
        print('Cannot find target folder')
        exit()
    if not os.path.exists(os.path.join('./Experiments', exp_folder, 'config.py')):
        print('Cannot find config.py in target folder')
        exit()

    exec('from Experiments.%s.config import configs' % exp_folder, globals())

    if device_idx == -1:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    tfconfig = tf.ConfigProto(allow_soft_placement = True, log_device_placement = False)
    tfconfig.gpu_options.allow_growth = True
    os.environ["CUDA_VISIBLE_DEVICES"] = device_idx
    sess = tf.Session(config = tfconfig)
    agentA = Agent(sess, configs, 'A', True, True, False)
    agentB = Agent(sess, configs, 'B', True, True, False)
    agents = [agentA, agentB]
    init = tf.global_variables_initializer()
    sess.run(init)

    game = Calendar(agentA.name_, agentB.name_, configs[1].step_reward,
                   configs[1].fail_reward, configs[1].success_reward, configs[1].num_slots,
                   '/home/luyao/Datasets/Calendar/calendar_%d_%d' % (configs[1].num_slots, configs[0].splits))

    switch_iters = configs[0].switch_iters if configs[0].get('switch_iters') else 3e4
    ckpt_start = 75000
    ckpt_interval = 75000
    num_ckpts = 20
    ckpt_iters = [ckpt_start + i * ckpt_interval for i in range(num_ckpts)]
    ckpt_dir = os.path.join('Experiments', exp_folder, configs[0].ckpt_dir)
    accuracies = []
    for ckpt_iter in ckpt_iters:
        fa = open(os.path.join(ckpt_dir, 'A', 'checkpoint'), 'w+')
        fb = open(os.path.join(ckpt_dir, 'B', 'checkpoint'), 'w+')
        a_iter = ckpt_iter if int((ckpt_iter - 1) / switch_iters) % 2 == 0 else int((ckpt_iter - 1) / switch_iters) * switch_iters
        b_iter = ckpt_iter if int((ckpt_iter - 1) / switch_iters) % 2 == 1 else int((ckpt_iter - 1) / switch_iters) * switch_iters
        fa.write('model_checkpoint_path: "checkpoint-%d"\nall_model_checkpoint_paths: "checkpoint-%d"' % (a_iter, a_iter))
        fb.write('model_checkpoint_path: "checkpoint-%d"\nall_model_checkpoint_paths: "checkpoint-%d"' % (b_iter, b_iter))
        fa.close()
        fb.close()
        agentA.restore_ckpt(ckpt_dir)
        if b_iter > 0:
            agentB.restore_ckpt(ckpt_dir)
        np.random.seed(7)
        test_iters = int(6e3)
        records = []
        total_rewards = []
        for tit in tqdm(range(1, test_iters + 1)):
            epsilons = [0, 0]
            sar_sequence = play(agentA, agentB, game, epsilons, False)
            total_reward = np.sum([step[1] for step in sar_sequence[1:]])
            total_rewards.append(total_reward)
            success = (sar_sequence[-1][1] == configs[1].success_reward)
            records.append(success)
        print('%d: %f' % (ckpt_iter, np.mean(records)))
        accuracies.append(np.mean(records))
        #print('average successful rate: %f, average reward: %f' % (np.mean(records), np.mean(total_rewards)))
    f = open(os.path.join('Experiments', exp_folder, 'results.txt'), 'w+')
    for acc in accuracies:
        f.write('%f\n' % acc)
    return

if __name__ == '__main__':
    main()
