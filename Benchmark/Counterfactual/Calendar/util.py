import tensorflow as tf
import numpy as np

def change_pair_test(config, p1, p2, calendar):
    null_action = np.zeros([config.action_encode_dim])
    null_action_p_idx = np.zeros([config.action_encode_dim + 2])
    null_in_x = np.zeros([config.num_slot + config.action_encode_dim + 2 + 2])
    p1_idx_oh = np.array([0, 1])
    p2_idx_oh = np.array([1, 0])
    all_action_encode = calendar.action_tensor_
    player1 = 0 # teacher
    player2 = 1 # student

    succeed = []
    msg_succeed = []
    action_succeed = []
    traj_len = []
    p2_wrong_count = 0

    for _ in range(config.test_batch_size):
        calendar.reset(train = False)
        p1_calendar = calendar.start()[player1].private
        p2_calendar = calendar.start()[player2].private
        
        # player 1 & 2 act turn by turn
        terminate = False
        t = 0
        fst_act_player = np.random.randint(0, 2)
        
        in_state = [None, None]
        prev_action_p_idx = null_action_p_idx
        
        while not terminate:
            if t >= config.terminate_len:
                succeed.append(False)
                if np.random.random() < 0.5:
                    p2_wrong_count += 1
                traj_len.append(t)
                break

            else:
                in_x_1 = np.concatenate([p1_calendar, prev_action_p_idx, p1_idx_oh])
                in_x_2 = np.concatenate([p2_calendar, prev_action_p_idx, p2_idx_oh])
                action_1, action_prob_1, state_1 = p1.sample_action(in_x_1, in_state[0])
                action_2, action_prob_2, state_2 = p2.sample_action(in_x_2, in_state[1])
                in_state = [state_1, state_2]

                if t % 2 == fst_act_player:
                    env_feedback = calendar.proceed({player1: action_1})[player1]
                    reward = env_feedback.reward
                    terminate = env_feedback.terminate

                    if 'msg_success' in env_feedback:
                        msg_succeed.append(env_feedback.msg_success)
                    
                    if terminate:
                        if 'action_success' in env_feedback:
                            succeed.append(env_feedback.action_success)
                            action_succeed.append(env_feedback.action_success)
                        else:
                            succeed.append(False)
                        traj_len.append(t + 1)
                    
                    prev_action_p_idx = np.concatenate([all_action_encode[action_1], p1_idx_oh])

                else:
                    env_feedback = calendar.proceed({player2: action_2})[player2]
                    reward = env_feedback.reward
                    terminate = env_feedback.terminate

                    if 'msg_success' in env_feedback:
                        msg_succeed.append(env_feedback.msg_success)

                    if terminate:
                        if 'action_success' in env_feedback:
                            succ = env_feedback.action_success
                            succeed.append(succ)
                            action_succeed.append(succ)
                        else:
                            succ = False
                            succeed.append(False)
                        traj_len.append(t + 1)
                        if not succ:
                            p2_wrong_count += 1

                    prev_action_p_idx = np.concatenate([all_action_encode[action_2], p2_idx_oh])

            t += 1

    max_traj_len = max(traj_len)
    mean_traj_len = np.mean(traj_len)
    accuracy = np.mean(succeed)
    wrong_count = len(succeed) - np.sum(succeed)
    p2_wrong = p2_wrong_count / wrong_count
    msg_acc = np.mean(msg_succeed)
    action_acc = np.mean(action_succeed)

    print('[change_pair_test]: ', accuracy)
    print('traj_len: ', mean_traj_len)


def max_len_msg_test(config, policy, calendar):
    null_action = np.zeros([config.action_encode_dim])
    null_action_p_idx = np.zeros([config.action_encode_dim + 2])
    null_in_x = np.zeros([config.num_slot + config.action_encode_dim + 2 + 2])
    p1_idx_oh = np.array([0, 1])
    p2_idx_oh = np.array([1, 0])
    all_action_encode = calendar.action_tensor_
    player1 = 0 # teacher
    player2 = 1 # student

    succeed = []
    msg_succeed = []
    action_succeed = []
    traj_len = []
    p2_wrong_count = 0

    max_len_msg_sent_l = []
    max_len_msg_l = []
    msg_len_diff_l = []

    for _ in range(config.test_batch_size):
        calendar.reset(train = False)
        p1_calendar = calendar.start()[player1].private
        p2_calendar = calendar.start()[player2].private
        p1_cid = calendar.player1_cid_
        p2_cid = calendar.player2_cid_
        p1_msg_max_len = max([len(msg) for msg in calendar.get_msg(p1_cid)[0]])
        p2_msg_max_len = max([len(msg) for msg in calendar.get_msg(p2_cid)[0]])
        
        # player 1 & 2 act turn by turn
        terminate = False
        t = 0
        fst_act_player = np.random.randint(0, 2)
        
        in_state = [None, None]
        prev_action_p_idx = null_action_p_idx
        
        while not terminate:
            if t >= 2:
                break

            else:
                in_x_1 = np.concatenate([p1_calendar, prev_action_p_idx, p1_idx_oh])
                in_x_2 = np.concatenate([p2_calendar, prev_action_p_idx, p2_idx_oh])
                action_1, action_prob_1, state_1 = policy.sample_action(in_x_1, in_state[0])
                action_2, action_prob_2, state_2 = policy.sample_action(in_x_2, in_state[1])
                in_state = [state_1, state_2]

                if t % 2 == fst_act_player:
                    env_feedback = calendar.proceed({player1: action_1})[player1]
                    reward = env_feedback.reward
                    terminate = env_feedback.terminate

                    if 'msg_success' in env_feedback:
                        msg_succ = env_feedback.msg_success
                        msg_succeed.append(msg_succ)
                        if msg_succ:
                            msg_encode = all_action_encode[action_1]
                            msg_len = len(np.where(msg_encode[:config.num_slot])[0])
                            max_len_msg_sent_l.append(msg_len == p1_msg_max_len)
                            max_len_msg_l.append(p1_msg_max_len)
                            msg_len_diff = p1_msg_max_len - msg_len
                            assert(msg_len_diff >= 0)
                            msg_len_diff_l.append(msg_len_diff)
                                
                    
                    if terminate:
                        if 'action_success' in env_feedback:
                            succeed.append(env_feedback.action_success)
                            action_succeed.append(env_feedback.action_success)
                        else:
                            succeed.append(False)
                        traj_len.append(t + 1)
                    
                    prev_action_p_idx = np.concatenate([all_action_encode[action_1], p1_idx_oh])

                else:
                    env_feedback = calendar.proceed({player2: action_2})[player2]
                    reward = env_feedback.reward
                    terminate = env_feedback.terminate

                    if 'msg_success' in env_feedback:
                        msg_succ = env_feedback.msg_success
                        msg_succeed.append(msg_succ)
                        if msg_succ:
                            msg_encode = all_action_encode[action_2]
                            msg_len = len(np.where(msg_encode[:config.num_slot])[0])
                            max_len_msg_sent_l.append(msg_len == p2_msg_max_len)
                            max_len_msg_l.append(p2_msg_max_len)
                            msg_len_diff = p2_msg_max_len - msg_len
                            assert(msg_len_diff >= 0)
                            msg_len_diff_l.append(msg_len_diff)

                    if terminate:
                        if 'action_success' in env_feedback:
                            succ = env_feedback.action_success
                            succeed.append(succ)
                            action_succeed.append(succ)
                        else:
                            succ = False
                            succeed.append(False)
                        traj_len.append(t + 1)
                        if not succ:
                            p2_wrong_count += 1

                    prev_action_p_idx = np.concatenate([all_action_encode[action_2], p2_idx_oh])

            t += 1

    # max_traj_len = max(traj_len)
    # accuracy = np.mean(succeed)
    # wrong_count = len(succeed) - np.sum(succeed)
    # p2_wrong = p2_wrong_count / wrong_count
    # msg_acc = np.mean(msg_succeed)
    # action_acc = np.mean(action_succeed)
    

    print('[sent max msg len]: ', np.mean(max_len_msg_sent_l))
    print('[max msg len]: ', np.mean(max_len_msg_l))
    print('[diff]: ', np.mean(msg_len_diff_l))
