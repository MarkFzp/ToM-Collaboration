import sys
import os
import numpy as np
import tensorflow as tf

from interact import play, train_step
from Agent.agent import Agent
from Game.kitchen import Kitchen

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

    # if train_config.device.find('cpu') == 1:
    #     os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    tfconfig = tf.ConfigProto(allow_soft_placement = True, log_device_placement = False)
    tfconfig.gpu_options.allow_growth = True
    os.environ["CUDA_VISIBLE_DEVICES"] = device_idx
    sess = tf.Session(config = tfconfig)
    agentA = Agent(sess, configs, 'A', True, False, True)
    agentB = Agent(sess, configs, 'B', False, True, True)
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

    training_iters = int(1.6e6)
    training_iter_start = int(0) + 1
    switch_iters = 5e4
    check_frequency = 2000
    save_frequency = 10000
    update_q_target_frequency = 100
    agent_in_turn = 0
    epsilon_min = 0.05
    epsilon_start = 0.95
    epsilon_decay = switch_iters / 3
    records = []
    total_rewards = []
    q_losses = []
    p_losses = []
    b_losses = []
    for tit in range(training_iter_start, training_iters + 1):
        epsilon = epsilon_min + (epsilon_start - epsilon_min) * np.exp (-1 * (tit % switch_iters) / epsilon_decay)
        epsilons = [0, 0]
        epsilons[agent_in_turn] = epsilon
        sar_sequence = play(agentA, agentB, game, epsilons)
        total_reward = np.sum([step[2] for step in sar_sequence[1:]])
        total_rewards.append(total_reward)
        agentA.collect(sar_sequence)
        agentB.collect(sar_sequence)
        if len(agents[agent_in_turn].replay_buffer_) >= agents[agent_in_turn].training_config_.batch_size:
            q_loss, policy_loss, belief_loss = train_step(agents[agent_in_turn], tit, switch_iters)
            q_losses.append(q_loss)
            p_losses.append(policy_loss)
            if belief_loss is None:
                b_losses.append(998)
            else:    
                b_losses.append(belief_loss)
        records.append(sar_sequence[-1][2] == (configs[1].success_reward + configs[1].step_reward))
        if tit % update_q_target_frequency == 0:
            agents[agent_in_turn].target_net_update()
        if tit % check_frequency == 0:
            print('[%d:%d]epsilon: %f, average successful rate: %f,'
                    'average reward: %f, average q loss: %f, average p loss: %f, average b loss: %f'\
                  % (tit, agent_in_turn, epsilon, np.mean(records), np.mean(total_rewards),
                     np.mean(q_losses), np.mean(p_losses), np.mean(b_losses)))
            records = []
            total_rewards = []
            q_losses = []
            p_losses = []
            b_losses = []
        if tit % save_frequency == 0:
            agents[agent_in_turn].save_ckpt(ckpt_dir, tit)
        if tit % switch_iters == 0:
            agent_in_turn = (agent_in_turn + 1) % 2

    return

if __name__ == '__main__':
    main()
