import tensorflow as tf
import numpy as np
import os
import sys
sys.path.append("../..")
import logging
from copy import deepcopy
from shutil import copyfile
from QnetC import QNet
from AgentC import Agent
from replay_buffer import ReplayBuffer
from configC2 import config
from calendar import Calendar


def buffer_tranjectory(sess, student, teacher, calendar, buffer, qnet_primary, epsilon, iteration):
    calendar.reset()

    no_msg = np.zeros([config.num_slot])
    game = calendar.start()
    meetable = np.zeros(config.num_slot)
    meetable += calendar.meetable_slots_

    fingerprint = [epsilon, iteration / config.iteration]
    t_index = np.array([0])
    s_index = np.array([1])
    t_calendar = game['teacher']['private']
    s_calendar = game['student']['private']
    t_info = s_info = no_msg
    t_obv = np.concatenate([t_calendar, t_info])
    s_obv = np.concatenate([s_calendar, s_info])

    t_hidden_states = np.zeros([1, 1, config.in_x_dim])
    s_hidden_states = np.zeros([1, 1, config.in_x_dim])
    t_in_state = np.zeros([1, config.hidden_dim])
    s_in_state = np.zeros([1, config.hidden_dim])
    # player_index = []

    terminal = False
    t = 0
    t_reward = s_reward = 0
    total_rewards = 0
    t_tranjectory = []
    s_tranjectory = []
    sar_sequence = [] #[(calendarA, calendarB, meetable_slots, Null-action, first_actor), ([a, ai], r, m), ([a, ai], r, m), ...]

    rd = np.random.choice(2, 1)
    starter = 'teacher' if rd is 0 else 'student'
    later = 'student' if starter is 'teacher' else 'teacher'
    starter_id = int(rd)

    last_action_embedding = np.zeros([1, 1, calendar.num_slots_ * 2])
    sar_embedding_sequence = [(np.expand_dims(np.expand_dims(t_calendar, 0), 0),
                               np.expand_dims(np.expand_dims(s_calendar, 0), 0),
                               last_action_embedding, meetable, starter_id)]
    sar_sequence.append(sar_embedding_sequence)

    # for t in range(config.max_time):
    while not terminal:

        # teacher's turn
        if t % 2 == 0 or t == 0:
            player = starter
            other = later
            t_obv = np.expand_dims(np.expand_dims(t_obv, 0), 0)
            q_values = qnet_primary.get_q(t_obv, [[fingerprint]], t_hidden_states, [t_index])
            t_action, t_action_chosen = teacher.choose_action(sess, q_values, epsilon)
            env_feedback = calendar.proceed({player: t_action_chosen})
            t_reward = env_feedback[player].reward
            terminal = env_feedback[player].terminate
            s_info = no_msg if env_feedback[other].observation is None else env_feedback[other].observation

            t_tranjectory.append([t_calendar, t_obv, t_hidden_states, t_index, t_action_chosen, t_reward, terminal, fingerprint])
            total_rewards += t_reward

            t_hidden_states, t_in_state = qnet_primary.get_hidden_states(sess, t_obv, t_action, t_in_state)
            s_obv = np.concatenate([s_calendar, s_info])
            # t_obv = np.concatenate([kitchen.observe('teacher').menu_embed, np.transpose(target)], axis=1)

            last_action_embedding = np.expand_dims(calendar.action_tensor_[int(t_action_chosen): int(t_action_chosen) + 1, :], 0)
            oh_last_action_index = np.zeros([1, 1, calendar.num_actions_])
            np.put(oh_last_action_index, int(t_action_chosen), 1)
            sar_sequence.append([[last_action_embedding, oh_last_action_index], t_reward])

        # student's turn:
        else:
            player = starter
            other = later
            s_obv = np.expand_dims(np.expand_dims(s_obv, 0),0)
            q_values = qnet_primary.get_q(s_obv, [[fingerprint]], s_hidden_states, [s_index])
            s_action, s_action_chosen = student.choose_action(sess, q_values, epsilon)
            env_feedback = calendar.proceed({player: s_action_chosen})
            s_reward = env_feedback[player].reward
            terminal = env_feedback[player].terminate
            t_info = no_msg if env_feedback[other].observation is None else env_feedback[other].observation

            s_tranjectory.append([s_calendar, s_obv, s_hidden_states, s_index, s_action_chosen, s_reward, terminal, fingerprint])
            total_rewards += s_reward

            s_hidden_states, s_in_state = qnet_primary.get_hidden_states(sess, s_obv, s_action, s_in_state)
            t_obv = np.concatenate([t_calendar, t_info])

            last_action_embedding = np.expand_dims(calendar.action_tensor_[int(s_action_chosen): int(s_action_chosen) + 1, :], 0)
            oh_last_action_index = np.zeros([1, 1, calendar.num_actions_])
            np.put(oh_last_action_index, int(s_action_chosen), 1)
            sar_sequence.append([[last_action_embedding, oh_last_action_index], s_reward])

        t += 1

    success = True if (s_reward is 1 or t_reward is 1) else False
    if starter_id == 0:
        buffer.store(t_tranjectory, s_tranjectory, meetable, success, starter_id, sar_sequence)
    else:
        buffer.store(s_tranjectory, t_tranjectory, meetable, success, starter_id, sar_sequence)
    return total_rewards, t_tranjectory, s_tranjectory, success, starter_id, sar_sequence


def train(buffer, teacher, student, qnet_primary, qnet_target, episode):

    samples_index = np.random.choice(buffer.size, config.batch_size, replace=False)
    t_samples = [buffer.get_data(i, 0) for i in samples_index]
    s_samples = [buffer.get_data(i, 1) for i in samples_index]
    meeatable = [buffer.get_meetable(i) for i in samples_index]
    sar_sequence = [buffer.get_sar_sequence(i) for i in samples_index]
    starters = [buffer.get_starter(i) for i in samples_index]
    t_samples_length = [len(t_samples[i]) for i in range(len(t_samples))]
    s_samples_length = [len(s_samples[i]) for i in range(len(s_samples))]
    t_max_length = np.max(t_samples_length)
    s_max_length = np.max(s_samples_length)

    t_loss = train_data(0, teacher, t_samples, s_samples, starters, sar_sequence, t_max_length, qnet_primary, qnet_target, episode)
    s_loss = train_data(1, student, s_samples, t_samples, starters, sar_sequence, s_max_length, qnet_primary, qnet_target, episode)

    return t_loss + s_loss, np.mean(t_samples_length), np.mean(s_samples_length)


def train_data(agent_id, agent, samples, samples2, starters, sar_sequence, max_length, qnet_primary, qnet_target, episode):

    b_calendar = np.zeros([config.batch_size, max_length, config.num_slot])
    b_obv = np.zeros([config.batch_size, max_length, config.num_obv])
    b_hidden_states = np.zeros([config.batch_size, max_length, config.out_x_dim])
    b_player_index = np.zeros([config.batch_size, max_length, 1])
    b_action_chosen = np.zeros([config.batch_size, max_length, 1])
    b_reward = np.zeros([config.batch_size, max_length, 1])
    b_terminal = np.zeros([config.batch_size, max_length, 1])
    b_fingerprint = np.zeros([config.batch_size, max_length, config.num_fingerprint])
    o_reward = np.zeros([config.batch_size, max_length, 1])
    o_terminal = np.zeros([config.batch_size, max_length, 1])

    for i in range(len(samples)):
        for j in range(len(samples[i])):
            b_calendar[i][j] = samples[i][j][0]
            b_obv[i][j] = samples[i][j][1]
            b_hidden_states[i][j] = samples[i][j][2]
            b_player_index[i][j] = samples[i][j][3]
            b_action_chosen[i][j] = samples[i][j][4]
            b_reward[i][j] = samples[i][j][5]
            b_terminal[i][j] = samples[i][j][6]
            b_fingerprint[i][j] = samples[i][j][7]

    for i in range(len(samples2)):
        if agent_id != starters[i]:
            for j in range(len(samples2[i]) - 1):
                o_reward[i][j] = samples2[i][j+1][5]
                o_terminal[i][j] = samples2[i][j+1][6]
        else:
            for j in range(len(samples2[i])):
                o_reward[i][j] = samples2[i][j][5]
                o_terminal[i][j] = samples2[i][j][6]

    # b_q_values = np.zeros([config.batch_size, max_length, config.num_action])
    b_q_mask = np.zeros([config.batch_size, max_length, config.num_action])
    b_target_qs = np.zeros([config.batch_size, max_length])

    # for i in reversed(range(max_length)):
    #     # b_hs = np.expand_dims(b_hidden_states[:, i], 1)
    #     q_value = qnet_target.get_q(b_obv[:, i], b_hs, b_player_index[:, i, 0])
    #     b_q_values[:, i] = q_value

    b_q_values = qnet_target.get_q(b_obv, b_fingerprint, b_hidden_states, b_player_index[:,:,0])
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
    loss, q_predict = qnet_primary.fit_target_q(b_obv, b_fingerprint, b_hidden_states, b_player_index[:, :, 0],
                                b_action_chosen[:, :, 0], b_target_qs, b_q_mask)

    if episode % 1000 == 0:
        # if slot_belief is not None:
        #    pdb.set_trace()
        valid_msg_q_spvs = []
        invalid_msg_q_spvs = []
        valid_prop_q_spvs = []
        invalid_prop_q_spvs = []
        related_indices = np.nonzero(b_q_mask)
        for i in range(related_indices[0].shape[0]):
            j = related_indices[1][i]
            if related_indices[2][i] < agent.calendar_.num_msgs_:
                valid = (np.sum((sar_sequence[related_indices[0][i]][0][0][agent_id] - \
                                 agent.calendar_.action_tensor_[related_indices[2][i],
                                 0: agent.calendar_.num_slots_]) == -1) == 0)
                if valid:
                    valid_msg_q_spvs.append(b_target_qs[related_indices[0][i], j])
                else:
                    invalid_msg_q_spvs.append(b_target_qs[related_indices[0][i], j])
            else:
                valid = (np.sum(
                    (1 - agent.calendar_.action_tensor_[related_indices[2][i], agent.calendar_.num_slots_:]) * \
                    sar_sequence[related_indices[0][i]][0][0][agent_id]) == 0)
                if valid:
                    valid_prop_q_spvs.append(b_target_qs[related_indices[0][i], j])
                else:
                    invalid_prop_q_spvs.append(b_target_qs[related_indices[0][i], j])

        logging.info('valid msg q_spvs: %f' % np.mean(valid_msg_q_spvs))
        logging.info('invalid msg q_spvs: %f' % np.mean(invalid_msg_q_spvs))
        logging.info('valid prop q_spvs: %f' % np.mean(valid_prop_q_spvs))
        logging.info('invalid prop q_spvs: %f' % np.mean(invalid_prop_q_spvs))
        right = np.zeros(2)
        diffs1 = []
        diffs2 = []
        agent2_counter = np.zeros(3)
        agent2_counter2 = np.zeros(3)
        lengths = np.zeros(config.batch_size)
        last_act = np.zeros(2)
        all_msgs_taken = set()
        all_props_taken = set()
        for i in range(config.batch_size):
            lengths[i] = len(sar_sequence[i])
            last_act_idx = int(np.nonzero(sar_sequence[i][-1][0][1])[-1])
            last_act[int(last_act_idx >= agent.calendar_.num_msgs_)] += 1
            calendar_id = int(''.join(str(sar_sequence[i][0][0][agent_id][0, 0, :])[1: -1].split()), 2)
            valid_msgs = agent.calendar_.get_msg(calendar_id)[0]
            valid_msg_idx = [agent.calendar_.all_msgs_.index(vm) for vm in valid_msgs]
            valid_mask = np.zeros(agent.calendar_.num_msgs_)
            slot_mask = np.squeeze(1 - sar_sequence[i][0][0][agent_id])
            np.put(valid_mask, valid_msg_idx, 1)
            if np.sum(valid_mask) == agent.calendar_.num_msgs_:
                continue
            valid_msg_average_q = np.sum(q_predict[i, 0,
                                         0: agent.calendar_.num_msgs_] * valid_mask) / np.sum(valid_mask)
            valid_propose_average_q = np.sum(q_predict[i, 0,
                                             agent.calendar_.num_msgs_: -1] * slot_mask) / np.sum(slot_mask)
            invalid_valid_propose_average_q = np.sum(
                q_predict[i, 0, agent.calendar_.num_msgs_: -1] * (
                            1 - slot_mask)) / np.sum(1 - slot_mask)
            invalid_msg_average_q = np.sum(
                q_predict[i, 0, 0: agent.calendar_.num_msgs_] * (
                            1 - valid_mask)) / np.sum(1 - valid_mask)
            # diffs1.append(valid_msg_average_q - valid_propose_average_q)
            diffs1.append(valid_propose_average_q - invalid_valid_propose_average_q)
            diffs2.append(valid_msg_average_q - invalid_msg_average_q)
            if valid_propose_average_q > invalid_valid_propose_average_q:
                right[0] += 1
            if valid_msg_average_q > invalid_msg_average_q:
                right[1] += 1
            # check helping agents actions
            other_turn = int(1 - agent_id)
            if len(sar_sequence[i]) - 1 > other_turn:
                agent2_counter[2] += 1
                action_idx = int(np.nonzero(sar_sequence[i][1 + other_turn][0][1])[-1])
                if action_idx >= agent.calendar_.num_msgs_:
                    all_props_taken.add(action_idx)
                    agent2_counter[1] += 1
                    valid_self = (np.sum((1 - sar_sequence[i][1 + other_turn][0][0][0, 0, agent.calendar_.num_slots_:]) *
                                         sar_sequence[i][0][0][1 - agent_id]) == 0)
                    agent2_counter[0] += valid_self
                else:
                    all_msgs_taken.add(action_idx)
                    agent2_counter2[2] += 1
                    valid_self = (np.sum((sar_sequence[i][0][0][1 - agent_id] - sar_sequence[i][1 + other_turn][0][0][
                                                                                       0, 0,
                                                                                       0: agent.calendar_.num_slots_]) == -1) == 0)
                    agent2_counter2[1] += valid_self
                    msg_indices = np.where(agent.calendar_.action_tensor_[action_idx, :][0: agent.calendar_.num_slots_])
                    left_slot = sar_sequence[i][0][0][0][0, 0, msg_indices[0][0] - 1] if msg_indices[0][0] > 0 else 0
                    right_slot = sar_sequence[i][0][0][1 - agent_id][0, 0, msg_indices[0][-1] + 1] if msg_indices[0][
                                                                                                           -1] < agent.calendar_.num_slots_ - 1 else 0
                    agent2_counter2[0] += (left_slot + right_slot == 0) * valid_self
                    # if left_slot + right_slot != 0:
                    #    print('(short)', action_idx, sar_sequence[i][0][1 - agent.index_][0, 0, ...])
                    # else:
                    #    print('(long)', action_idx, sar_sequence[i][0][1 - agent.index_][0, 0, ...])
        logging.info('agent helping takes %d different messages, %d different proposals' % (
        len(all_msgs_taken), len(all_props_taken)))
        logging.info('agent helping propose: %f, agent helping correct propose: %f' % (agent2_counter[1] / agent2_counter[2],
                                                                                agent2_counter[0] / agent2_counter[1]))
        logging.info('agent helping msg: %f, agent helping correct msg: %f, longest interval: %f' %
              (agent2_counter2[2] / agent2_counter[2],
               agent2_counter2[1] / agent2_counter2[2],
               agent2_counter2[0] / agent2_counter2[1]))
        logging.info('batch average length: %f, end with msg: %f, end with propose: %f' % (np.mean(lengths),
                                                                                    last_act[
                                                                                        0] / config.batch_size,
                                                                                    last_act[
                                                                                        1] / config.batch_size))
        logging.info(
            '<%d: %d>batch valid propose > invalid propose ratio: %f, mean diff: %f, valid msg > invalid msg ratio: %f, mean diff: %f' \
            % (episode, agent_id, right[0] / config.batch_size, np.mean(diffs1),
               right[1] / config.batch_size, np.mean(diffs2)))

    # for i in range(len(samples)):
    #     for j in range(max_length):
    #         print('%f ' % b_target_qs[i][j])
    #     print('\n')

    return loss


def main():
    if not config.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    sess_config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    sess_config.gpu_options.allow_growth = True
    sess = tf.Session(config=sess_config)
    # sess = tf.Session()

    student = Agent(config, 'student')
    teacher = Agent(config, 'teacher')

    calendar = Calendar('teacher', 'student', config.step_reward, config.fail_reward,
                        config.succeed_reward, config.num_slot, config.dataset)

    qnet_primary = QNet(config, sess, False)
    qnet_target = QNet(config, sess, True)
    buffer = ReplayBuffer(config)

    saver = tf.train.Saver(max_to_keep=None)
    if not os.path.exists(config.ckpt_dir):
        os.makedirs(config.ckpt_dir)

    ckpt_status = tf.train.get_checkpoint_state(config.ckpt_dir)
    if ckpt_status:
        saver.restore(sess, ckpt_status.model_checkpoint_path)
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
    msgs_success = []
    actions_success = []

    for episode in range(config.start_iter, config.iteration + 1):

        epsilon = config.epsilon_min + (config.epsilon_start - config.epsilon_min) \
                * np.exp(-1 * (episode % config.switch_iters) / config.epsilon_decay)

        reward, t, s, success, starter_id, sar_sequence = buffer_tranjectory(sess, student, teacher,
                calendar, buffer, qnet_primary, epsilon, episode)
        total_rewards.append(reward)

        if buffer.size >= config.batch_size:
            q_loss, t_len, s_len = train(buffer, teacher, student, qnet_primary, qnet_target, episode)
            q_losses.append(q_loss)
            t_lens.append(t_len)
            s_lens.append(s_len)

        if episode % config.update_q_target_frequency == 0:
            # qnet_target.target_net_update(q_var, q_target_var)
            qnet_target.target_net_update()

        if episode % config.print_iteration == 0:
            accuracy = buffer.accuracy()
            # print('[%d] epsilon: %f, average reward: %f, average loss: %f, accuracy: %f, t_len: %f, s_len %f'
            #       % (episode, epsilon, np.mean(total_rewards), np.mean(q_losses), accuracy, np.mean(t_lens), np.mean(s_lens)))

            logging.info('[%d] epsilon: %f, average reward: %f, average loss: %f, accuracy: %f, t_len: %f, s_len %f'
                  % (episode, epsilon, np.mean(total_rewards), np.mean(q_losses), accuracy, np.mean(t_lens), np.mean(s_lens)))

            total_rewards = []
            q_losses = []
            t_lens = []
            s_lens = []

        if episode % config.save_frequency == 0:
            saver.save(sess, os.path.join(config.ckpt_dir, 'checkpoint'), global_step=episode)
            print('ckpt %d saved' % episode)
            print()

        # if episode % switch_iters == 0:
        #     agent_in_turn = (agent_in_turn + 1) % 2


if __name__ == '__main__':
    main()
