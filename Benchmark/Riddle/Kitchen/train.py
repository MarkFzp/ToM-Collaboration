import tensorflow as tf
import numpy as np
import os
import sys
sys.path.append("../..")
import logging
from copy import deepcopy
from shutil import copyfile
from Qnet import QNet
from Agent import Agent
from replay_buffer import ReplayBuffer
from config import config
from kitchen import Kitchen
from EpsilonGreedy import EpsilonGreedy


def buffer_tranjectory(sess, student, teacher, kitchen, buffer, qnet_primary, epsilon, iteration):
    kitchen.reset()

    obv = kitchen.start()
    t_index = np.array([0])
    s_index = np.array([1])
    target = np.concatenate([obv['teacher']['private'], np.zeros([config.num_ingredient - config.num_candidate])])
    target_index = np.argmax(obv['teacher']['private'])
    target_contents = obv['teacher']['observation'].menu_embed[target_index]

    # obv: menu, target, workplace
    workplace = np.zeros([1, config.num_ingredient])
    t_know = np.concatenate([obv['teacher']['observation'].menu_embed, [target]])
    s_know = np.concatenate([obv['student']['observation'].menu_embed, np.zeros([1, config.num_ingredient])])
    t_obv = np.concatenate([t_know, workplace])
    s_obv = np.concatenate([s_know, workplace])

    t_hidden_states = np.zeros([1, 1, config.in_x_dim])
    s_hidden_states = np.zeros([1, 1, config.in_x_dim])
    t_in_state = np.zeros([1, config.hidden_dim])
    s_in_state = np.zeros([1, config.hidden_dim])
    # player_index = []

    terminal = False
    t = 0
    total_rewards = 0
    t_tranjectory = []
    s_tranjectory = []

    # for t in range(config.max_time):
    while not terminal:

        # teacher's turn
        if t % 2 == 0 or t == 0:
            # player_index.append(t_index)
            t_obv = np.expand_dims(np.reshape(t_obv, [-1, config.num_obv * config.num_ingredient]), 1)
            q_values = qnet_primary.get_q(t_obv, t_hidden_states, [t_index])
            t_action, t_action_chosen = teacher.choose_action(q_values, epsilon)
            env_feedback = kitchen.proceed({'teacher': t_action_chosen})['teacher']
            t_reward = env_feedback.reward
            terminal = env_feedback.terminate

            t_tranjectory.append([t_obv, t_hidden_states, t_index, t_action_chosen, t_reward, terminal, workplace])
            total_rewards += t_reward

            workplace = deepcopy(env_feedback.observation.workplace_embed)
            # t_action = np.expand_dims(t_action, axis=0)
            t_hidden_states, t_in_state = qnet_primary.get_hidden_states(sess, t_obv, t_action, t_in_state)
            t_obv = np.concatenate([t_know, [workplace]])
            s_obv = np.concatenate([s_know, [workplace]])
            # t_obv = np.concatenate([kitchen.observe('teacher').menu_embed, np.transpose(target)], axis=1)

        # student's turn:
        else:
            # player_index.append(s_index)
            s_obv = np.expand_dims(np.reshape(s_obv, [-1, config.num_obv * config.num_ingredient]), 1)
            q_values = qnet_primary.get_q(s_obv, s_hidden_states, [s_index])
            s_action, s_action_chosen = student.choose_action(q_values, epsilon)
            env_feedback = kitchen.proceed({'student': s_action_chosen})['student']
            s_reward = env_feedback.reward
            terminal = env_feedback.terminate

            s_tranjectory.append([s_obv, s_hidden_states, s_index, s_action_chosen, s_reward, terminal, workplace])
            total_rewards += s_reward

            workplace = env_feedback.observation.workplace_embed
            # s_action = np.expand_dims(s_action, axis=0)
            s_hidden_states, s_in_state = qnet_primary.get_hidden_states(sess, s_obv, s_action, s_in_state)
            t_obv = np.concatenate([t_know, [workplace]])
            s_obv = np.concatenate([s_know, [workplace]])

        t += 1

    # buffer.success.append(env_feedback.success)
    # buffer.store(t_tranjectory, 0)
    # buffer.store(s_tranjectory, 1)
    buffer.store(t_tranjectory, s_tranjectory, target_contents, env_feedback.success)
    return total_rewards, t_tranjectory, s_tranjectory, target_contents, env_feedback.success


def train(sess, buffer, qnet_primary, qnet_target, episode):

    samples_index = np.random.choice(buffer.size, config.batch_size, replace=False)
    t_samples = [buffer.get_data(i, 0) for i in samples_index]
    s_samples = [buffer.get_data(i, 1) for i in samples_index]
    targets = [buffer.get_target(i) for i in samples_index]
    t_samples_length = [len(t_samples[i]) for i in range(len(t_samples))]
    s_samples_length = [len(s_samples[i]) for i in range(len(s_samples))]
    t_max_length = np.max(t_samples_length)
    s_max_length = np.max(s_samples_length)

    t_loss = train_data(sess, 0, targets, t_samples, s_samples, t_max_length, qnet_primary, qnet_target, episode)
    s_loss = train_data(sess, 1, targets, s_samples, t_samples, s_max_length, qnet_primary, qnet_target, episode)

    return t_loss + s_loss, np.mean(t_samples_length), np.mean(s_samples_length)


def train_data(sess, agent, targets, samples, samples2, max_length, qnet_primary, qnet_target, episode):

    b_obv = np.zeros([config.batch_size, max_length, config.num_obv * config.num_ingredient])
    b_hidden_states = np.zeros([config.batch_size, max_length, config.out_x_dim])
    b_player_index = np.zeros([config.batch_size, max_length, 1])
    b_action_chosen = np.zeros([config.batch_size, max_length, 1])
    b_reward = np.zeros([config.batch_size, max_length, 1])
    b_terminal = np.zeros([config.batch_size, max_length, 1])
    b_workplace = np.zeros([config.batch_size, max_length, config.num_ingredient])
    o_reward = np.zeros([config.batch_size, max_length, 1])
    o_terminal = np.zeros([config.batch_size, max_length, 1])

    for i in range(len(samples)):
        for j in range(len(samples[i])):
            b_obv[i][j] = samples[i][j][0]
            b_hidden_states[i][j] = samples[i][j][1]
            b_player_index[i][j] = samples[i][j][2]
            b_action_chosen[i][j] = samples[i][j][3]
            b_reward[i][j] = samples[i][j][4]
            b_terminal[i][j] = samples[i][j][5]
            b_workplace[i][j] = samples[i][j][6]

    for i in range(len(samples2)):
        if agent == 1:
            for j in range(len(samples2[i]) - 1):
                o_reward[i][j] = samples2[i][j+1][4]
                o_terminal[i][j] = samples2[i][j+1][5]
        else:
            for j in range(len(samples2[i])):
                o_reward[i][j] = samples2[i][j][4]
                o_terminal[i][j] = samples2[i][j][5]

    # b_q_values = np.zeros([config.batch_size, max_length, config.num_action])
    b_q_mask = np.zeros([config.batch_size, max_length, config.num_action])
    b_target_qs = np.zeros([config.batch_size, max_length])

    # for i in reversed(range(max_length)):
    #     # b_hs = np.expand_dims(b_hidden_states[:, i], 1)
    #     q_value = qnet_target.get_q(b_obv[:, i], b_hs, b_player_index[:, i, 0])
    #     b_q_values[:, i] = q_value

    b_q_values = qnet_target.get_q(b_obv, b_hidden_states, b_player_index[:,:,0])
    max_target_qs = np.max(b_q_values, axis=2)
    # max_target_qs = max_target_qs[:, 1:]

    for j in reversed(range(max_length)):
        for i in range(len(samples)):
            action_index = int(b_action_chosen[i][j])
            b_q_mask[i, j, action_index] = 1 if j < len(samples[i]) else 0
            # b_q_mask[i, j] = 1 if j < len(samples[i]) else 0
            # max_target_qs[i][j] = max_target_qs[i][j] if j < len(samples[i]) - 1 else 0
            if b_terminal[i][j][0]:
                b_target_qs[i][j] = b_reward[i][j][0]
            elif o_terminal[i][j][0]:
                b_target_qs[i][j] = b_reward[i][j][0] + o_reward[i][j][0] * config.gamma
            elif j < len(samples[i]) - 1:
                b_target_qs[i][j] = b_reward[i][j][0] + o_reward[i][j][0] * config.gamma \
                                    + max_target_qs[i][j+1] * config.gamma * config.gamma

        # b_hs = np.expand_dims(b_hidden_states[:, j], 1)
        # q_loss = qnet_primary.fit_target_q(b_obv[:, j], b_hs, b_player_index[:, j, 0],
        #                             b_action_chosen[:, j, 0], b_target_qs[:, j], b_q_mask[:, j])
        # total_loss.append(q_loss)
    loss = qnet_primary.fit_target_q(b_obv, b_hidden_states, b_player_index[:, :, 0],
                                b_action_chosen[:, :, 0], b_target_qs, b_q_mask)

    # for i in range(len(samples)):
    #     for j in range(max_length):
    #         print('%f ' % b_target_qs[i][j])
    #     print('\n')

    if episode % config.print_iteration == 0:
        right = 0
        count = 0
        diffs = []
        for i in range(config.batch_size):
            if len(samples[i]) > 0:
                # target = b_obv[i][0][4 * config.num_ingredient:4 * config.num_ingredient + 4]
                # target_index = np.argmax(target)
                # target_menu = b_obv[i][0][target_index * config.num_ingredient:target_index * config.num_ingredient + config.num_ingredient]
                last_q_idx = len(samples[i]) - 1
                have = b_workplace[i][last_q_idx]
                needed = targets[i] - have > 0
                correct_average = np.sum(b_q_values[i, 0, :] * needed) / np.sum(needed)
                wrong_average = np.sum(b_q_values[i, 0, :] * (1 - needed)) / np.sum(1 - needed)
                diffs.append(correct_average - wrong_average)
                if correct_average > wrong_average:
                    right += 1
                count += 1
                # print('correct: %f, wrong: %f' % (correct_average, wrong_average))
        right_ratio = 1.0 * right / count if count > 0 else 0
        print('<%d: %d>batch correct > right ratio: %f, mean diff: %f' \
              % (episode, agent, right_ratio, np.mean(diffs)))

    return loss


def main():
    if not config.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    sess_config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    sess_config.gpu_options.allow_growth = True
    sess = tf.Session(config=sess_config)
    # sess = tf.Session()

    student = Agent(config, sess)
    teacher = Agent(config, sess)

    kitchen = Kitchen('teacher', 'student', \
                      config.step_reward, config.fail_reward, config.succeed_reward, \
                      range(config.num_attribute), config.num_candidate, config.max_attribute, config.dataset)

    qnet_primary = QNet(config, sess, False)
    qnet_target = QNet(config, sess, True)
    buffer = ReplayBuffer(config)

    saver = tf.train.Saver(max_to_keep=None)
    if not os.path.exists('CKPT'):
        os.makedirs('CKPT')

    ckpt_status = tf.train.get_checkpoint_state('./')
    if ckpt_status:
        saver.restore(sess, ckpt_status.model_checkpoint_path)
        print('Load model')
    else:
        print('ckpt_path does not exist')
        init = tf.global_variables_initializer()
        sess.run(init)

    if config.log:
        log_path = os.path.join('Log', config.log_file)
        if not os.path.exists('Log'):
            os.makedirs('Log')
        logger = logging.getLogger()
        logger.handlers = []
        file_handler = logging.FileHandler(log_path)
        console_handler = logging.StreamHandler(sys.stdout)
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        logger.setLevel(logging.INFO)

    total_rewards = []
    q_losses = []
    t_lens = []
    s_lens = []
    save_frequency = int((config.iteration - config.start_iter) / 20)

    for episode in range(config.start_iter + 1, config.iteration):

        if config.epsilon_exp:
            epsilon = config.epsilon_min + (config.epsilon_start - config.epsilon_min) \
                  * np.exp(-1 * (episode % config.switch_iters) / config.epsilon_decay)
        else:
            epsilon_greedy = EpsilonGreedy(config.num_action)
            epsilon = epsilon_greedy.get_epsilon(episode, config.training)

        reward, t, s, target, success = buffer_tranjectory(sess, student, teacher, kitchen, buffer, qnet_primary, epsilon, episode)
        total_rewards.append(reward)

        if buffer.size >= config.batch_size:
            q_loss, t_len, s_len = train(sess, buffer, qnet_primary, qnet_target, episode)
            q_losses.append(q_loss)
            t_lens.append(t_len)
            s_lens.append(s_len)

        if episode % config.update_q_target_frequency == 0:
            # qnet_target.target_net_update(q_var, q_target_var)
            qnet_target.target_net_update()

        if episode % config.print_iteration == 0:
            accuracy = buffer.accuracy()
            print('[%d] epsilon: %f, average reward: %f, average loss: %f, accuracy: %f, t_len: %f, s_len %f'
                  % (episode, epsilon, np.mean(total_rewards), np.mean(q_losses), accuracy, np.mean(t_lens), np.mean(s_lens)))

            logging.info('[%d] epsilon: %f, average reward: %f, average loss: %f, accuracy: %f, t_len: %f, s_len %f'
                  % (episode, epsilon, np.mean(total_rewards), np.mean(q_losses), accuracy, np.mean(t_lens), np.mean(s_lens)))

            total_rewards = []
            q_losses = []
            t_lens = []
            s_lens = []

        if episode % save_frequency == 0:
            saver.save(sess, config.ckpt_path, global_step=episode)
            print('ckpt saved')
            print()

        # if episode % switch_iters == 0:
        #     agent_in_turn = (agent_in_turn + 1) % 2


if __name__ == '__main__':
    main()
