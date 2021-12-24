import sys
import os
import numpy as np
import tensorflow as tf

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
    agentA = Agent(sess, configs, 'A', True, True, True)
    agentB = Agent(sess, configs, 'B', True, True, True)
    agents = [agentA, agentB]
    init = tf.global_variables_initializer()
    sess.run(init)
    ckpt_dir = os.path.join('Experiments', exp_folder, configs[0].ckpt_dir)
    with tf.device('/gpu:1'):
        agentA.restore_ckpt(ckpt_dir)
        agentB.restore_ckpt(ckpt_dir)

    game = Calendar(agentA.name_, agentB.name_, configs[1].step_reward,
                   configs[1].fail_reward, configs[1].success_reward, configs[1].num_slots,
                   '/home/luyao/Datasets/Calendar/calendar_%d_%d' % (configs[1].num_slots, configs[0].splits))

    training_iter_start = int(0) + 1
    switch_iters = configs[0].switch_iters if configs[0].get('switch_iters') else 3e4
    training_iters = int(2 * configs[1].num_slots * switch_iters)
    check_frequency = 2000
    save_frequency = int(1e5)#min(25000, training_iters / 50)
    update_q_target_frequency = 100
    agent_in_turn = 0
    epsilon_min = 0.05
    epsilon_start = 0.95
    epsilon_decay = switch_iters / 2
    records = []
    total_rewards = []
    q_losses = []
    p_losses = []
    b_losses = []
    cat_rewards = np.zeros(2)
    cat_counts = np.zeros(2)
    for tit in range(training_iter_start, training_iters + 1):
        epsilon = epsilon_min + (epsilon_start - epsilon_min) * np.exp (-1 * (tit % switch_iters) / epsilon_decay)
        epsilons = [0, 0]
        epsilons[agent_in_turn] = epsilon
        sar_sequence = play(agentA, agentB, game, epsilons)
        total_reward = np.sum([step[1] for step in sar_sequence[1:]])
        total_rewards.append(total_reward)
        action_idx = int(np.nonzero(sar_sequence[1][0][1])[-1])  
        cat_rewards[int(action_idx >= game.num_msgs_)] += total_reward
        cat_counts[int(action_idx >= game.num_msgs_)] += 1
        agentA.collect(sar_sequence)
        agentB.collect(sar_sequence)
        if len(agents[agent_in_turn].replay_buffer_) >= agents[agent_in_turn].training_config_.batch_size:
            q_loss, policy_loss, belief_loss = train_step(agents[agent_in_turn], tit, switch_iters)
            q_losses.append(q_loss)
            if policy_loss is None:
                p_losses.append(998)
            else:
                p_losses.append(policy_loss)
            if belief_loss is None:
                b_losses.append(998)
            else:    
                b_losses.append(belief_loss)
        records.append(sar_sequence[-1][1] == configs[1].success_reward)
        if tit % update_q_target_frequency == 0:
            agents[agent_in_turn].target_net_update()
        if tit % check_frequency == 0:
            print('[%d:%d]epsilon: %f, average successful rate: %f,'
                    'average reward: %f, average q loss: %f, average p loss: %f, average b loss: %f'\
                  % (tit, agent_in_turn, epsilon, np.mean(records), np.mean(total_rewards),
                     np.mean(q_losses), np.mean(p_losses), np.mean(b_losses)))
            print('\tmsg rewards VS propose rewards', cat_rewards / cat_counts)
            records = []
            total_rewards = []
            cat_rewards = np.zeros(2)
            cat_counts = np.zeros(2)
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
