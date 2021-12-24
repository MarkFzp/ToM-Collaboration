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
    ckpt_dir = os.path.join('Experiments', exp_folder, configs[0].ckpt_dir)
    agentA.restore_ckpt(ckpt_dir)
    agentB.restore_ckpt(ckpt_dir)

    game = Calendar(agentA.name_, agentB.name_, configs[1].step_reward,
                   configs[1].fail_reward, configs[1].success_reward, configs[1].num_slots,
                   '/home/luyao/Datasets/Calendar/calendar_%d_%d' % (configs[1].num_slots, configs[0].splits))

    np.random.seed(7)
    test_iters = int(6e3)
    records = []
    total_rewards = []
    errors = np.zeros(2)
    belief_correct = 0
    unique_id = [0, 0]
    have_unique_id = [0, 0]
    bad_A_error = 0
    rewards_UI_nonUI = [0, 0]
    use_UI_nonUI = [0, 0]
    cat_rewards = np.zeros(2)
    cat_counts = np.zeros(2)
    lengths = np.zeros(30)
    good_propose = np.zeros(2)
    propose = 0
    good_msg = np.zeros(2)
    msg = 0
    valid_success = 0
    prev_msg = [-1, -1]
    neighbor_propose = [0, 0]
    B_propose_A_msg = [0, 0]
    long_interval_reward = [0, 0] #measures next step rewards after CORRECT message intervals
    short_interval_reward = [0, 0]
    who2blame1 = defaultdict(int)
    who2blame2 = defaultdict(int)
    actions = defaultdict(int)
    for tit in tqdm(range(1, test_iters + 1)):
        epsilons = [0, 0]
        sar_sequence = play(agentA, agentB, game, epsilons, False)
        total_reward = np.sum([step[1] for step in sar_sequence[1:]])
        total_rewards.append(total_reward)
        success = (sar_sequence[-1][1] == configs[1].success_reward)
        first_actor_id = int(sar_sequence[0][4])
        first_action_idx = int(np.nonzero(sar_sequence[1][0][1])[-1])
        last_action_idx = int(np.nonzero(sar_sequence[-1][0][1])[-1])
        lengths[min(30, len(sar_sequence)) - 1] += 1
        records.append(success)
        for sid in range(1, len(sar_sequence)):
            actions[int(np.nonzero(sar_sequence[sid][0][1])[-1])] += 1
        # cat_rewards[int(first_action_idx >= game.num_msgs_)] += total_reward
        # cat_counts[int(first_action_idx >= game.num_msgs_)] += 1
        if len(sar_sequence) >= 3:
            if np.nonzero(sar_sequence[-2][0][1])[2][0] < game.num_msgs_ and\
               np.nonzero(sar_sequence[-2][0][1])[2][0] != 0:
                prev_msg[0] = np.nonzero(sar_sequence[-2][0][0])[2][0]
                prev_msg[-1] = np.nonzero(sar_sequence[-2][0][0])[2][-1]
            if np.nonzero(sar_sequence[-1][0][1])[2][0] >= game.num_msgs_:
                neighbor_propose[1] += 1
                propose_idx = np.nonzero(1 - sar_sequence[-1][0][0][0, 0, configs[1].num_slots:])[0]
                propose_idx = propose_idx[0] if len(propose_idx) == 1 else -1
                if propose_idx == prev_msg[0] - 1 or propose_idx == prev_msg[1] + 1:
                    neighbor_propose[0] += 1
        
        if last_action_idx >= game.num_msgs_:
            propose_idx = np.nonzero(1 - sar_sequence[-1][0][0][0, 0, configs[1].num_slots:])[0]
            propose_idx = propose_idx[0] if len(propose_idx) == 1 else -1
            B_propose_A_msg[1] += 1
            proposer_parity = (len(sar_sequence) - 1) % 2
            if not success and propose_idx != -1:
                for i in range(1, len(sar_sequence)):
                    if i % 2 != proposer_parity and np.nonzero(sar_sequence[i][0][1])[2][0] != 0:
                        start = np.nonzero(sar_sequence[i][0][0])[2][0]
                        end = np.nonzero(sar_sequence[i][0][0])[2][-1]
                        if propose_idx >= start and propose_idx <= end:
                            B_propose_A_msg[0] += 1

        if first_action_idx >= game.num_msgs_:
            propose += 1
            valid_self = (np.sum((1 - sar_sequence[1][0][0][0, 0, configs[1].num_slots: -1]) * sar_sequence[0][first_actor_id][0, 0, ...]) == 0)
            good_propose[int(first_action_idx == game.num_actions_ - 1)] += valid_self
            valid_success += int(valid_self) * int(success)
        else:
            msg += 1
            valid_self = (np.sum((sar_sequence[0][first_actor_id][0, 0, ...] - sar_sequence[1][0][0][0, 0, 0: configs[1].num_slots:]) == -1) == 0)
            good_msg[int(first_action_idx == 0)] += valid_self
            if valid_self:
                msg_indices = np.where(game.action_tensor_[first_action_idx, :][0: configs[1].num_slots])
                if len(msg_indices[0]) != 0:
                    left = sar_sequence[0][first_actor_id][0, 0, ...][msg_indices[0][0] - 1] if msg_indices[0][0] > 0 else 0
                    right = sar_sequence[0][first_actor_id][0, 0, ...][msg_indices[0][-1] + 1] if msg_indices[0][-1] < configs[1].num_slots - 1 else 0
                    if left + right == 0:
                        long_interval_reward[1] += 1
                        long_interval_reward[0] += (len(sar_sequence) - 1) * configs[1].step_reward - configs[1].fail_reward
                        if not success:
                            who2blame1['total'] += 1
                            if last_action_idx < game.num_msgs_ and len(sar_sequence) % 2 == 0:
                                who2blame1['%d_msg' % first_actor_id] += 1
                            elif last_action_idx < game.num_msgs_ and len(sar_sequence) % 2 == 1:
                                who2blame1['%d_msg' % first_actor_id] += 1
                            elif last_action_idx >= game.num_msgs_ and len(sar_sequence) % 2 == 0:
                                who2blame1['%d_prop' % first_actor_id] += 1
                            elif last_action_idx >= game.num_msgs_ and len(sar_sequence) % 2 == 1:
                                who2blame1['%d_prop' % first_actor_id] += 1
                    else:
                        short_interval_reward[1] += 1
                        short_interval_reward[0] += (len(sar_sequence) - 1) * configs[1].step_reward - configs[1].fail_reward
                        if not success:
                            who2blame2['total'] += 1
                            if last_action_idx < game.num_msgs_ and len(sar_sequence) % 2 == 0:
                                who2blame2['%d_msg' % first_actor_id] += 1
                            elif last_action_idx < game.num_msgs_ and len(sar_sequence) % 2 == 1:
                                who2blame2['%d_msg' % first_actor_id] += 1
                            elif last_action_idx >= game.num_msgs_ and len(sar_sequence) % 2 == 0:
                                who2blame2['%d_prop' % first_actor_id] += 1
                            elif last_action_idx >= game.num_msgs_ and len(sar_sequence) % 2 == 1:
                                who2blame2['%d_prop' % first_actor_id] += 1
                else:
                    short_interval_reward[1] += 1
                    short_interval_reward[0] += (len(sar_sequence) - 1) * configs[1].step_reward - configs[1].fail_reward
                    if not success:
                        who2blame2['total'] += 1
                        if last_action_idx < game.num_msgs_ and len(sar_sequence) % 2 == 0:
                            who2blame2['%d_msg' % first_actor_id] += 1
                        elif last_action_idx < game.num_msgs_ and len(sar_sequence) % 2 == 1:
                            who2blame2['%d_msg' % first_actor_id] += 1
                        elif last_action_idx >= game.num_msgs_ and len(sar_sequence) % 2 == 0:
                            who2blame2['%d_prop' % first_actor_id] += 1
                        elif last_action_idx >= game.num_msgs_ and len(sar_sequence) % 2 == 1:
                            who2blame2['%d_prop' % first_actor_id] += 1
        prev_msg = [-1, -1]
        #pdb.set_trace()
            
    print('average successful rate: %f, average reward: %f' % (np.mean(records), np.mean(total_rewards)))
    print(lengths / np.sum(lengths))
    print('first propose ratio:', 1.0 * propose / test_iters)
    print('first msg ratio:', 1.0 * msg / test_iters)
    print('valid propose, reject, propose:', 1.0 * good_propose / propose)
    print('valid msg no-0, 0:', 1.0 * good_msg / msg)
    print('valid propose success ratio:', 1.0 * valid_success / np.sum(good_propose))
    print('neighbor propose ratio: %f' % (neighbor_propose[0] / neighbor_propose[1]))
    print('propose bad partner msg ratio: %f' % (B_propose_A_msg[0] / B_propose_A_msg[1]))
    print('long interval success ratio: %f' % (long_interval_reward[0] / long_interval_reward[1]))
    print('short interval success ratio: %f' % (short_interval_reward[0] / short_interval_reward[1]))
    for k in who2blame1:
        if k != 'total':
            who2blame1[k] /= who2blame1['total']
    for k in who2blame2:
        if k != 'total':
            who2blame2[k] /= who2blame2['total']
    print(who2blame1)
    print(who2blame2)
    print(len(actions))
    brint(actions)

    return

if __name__ == '__main__':
    main()
