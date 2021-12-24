import numpy as np
from copy import deepcopy
from collections import Counter
from config import config

global player1, player2
player1 = 0
player2 = 1
global all_target_idx
all_target_idx = np.arange(config.num_candidate, dtype = np.int32)
global all_target
all_target = np.eye(config.num_candidate)
global uniform_belief
uniform_belief = np.ones(config.num_candidate) / config.num_candidate

def change_pair_test(calendar, policy_tea, policy_stu):
    succeed = []
    msg_succeed = []
    action_succeed = []
    traj_len = []
    p2_wrong_count = 0

    for _ in range(config.test_batch_size):
        calendar.reset(train = False)
        p1_cid = calendar.player1_cid_ - 1
        p2_cid = calendar.player2_cid_ - 1

        belief_1_1 = uniform_belief
        belief_1_2 = uniform_belief
        belief_2_1 = uniform_belief
        belief_2_2 = uniform_belief

        t = 0
        terminate = False
        while not terminate:
            if t >= config.terminate_len:
                succeed.append(False)
                if np.random.random() < 0.5:
                    p2_wrong_count += 1
                break

            elif t % 2 == 0:
                belief_2_1_stack = np.stack([belief_2_1] * config.num_candidate, axis = 0)
                belief_2_2_stack = np.stack([belief_2_2] * config.num_candidate, axis = 0)
                sampled_action_tea = policy_tea.sample_action(all_target, belief_2_1_stack)
                sampled_action_stu = policy_stu.sample_action(all_target, belief_2_2_stack)

                sampled_real_action = sampled_action_tea[p1_cid]
                belief_update_mask_tea = (sampled_action_tea == sampled_real_action)
                belief_update_mask_stu = (sampled_action_stu == sampled_real_action)
                
                new_belief_pre_norm_1 = belief_update_mask_tea * belief_1_1
                new_belief_pre_norm_sum_1 = np.sum(new_belief_pre_norm_1)
                assert(new_belief_pre_norm_sum_1 != 0)
                new_belief_1 = new_belief_pre_norm_1 / new_belief_pre_norm_sum_1
                
                new_belief_pre_norm_2 = belief_update_mask_stu * belief_1_2
                new_belief_pre_norm_sum_2 = np.sum(new_belief_pre_norm_2)
                if new_belief_pre_norm_sum_2 == 0:
                    new_belief_2 = uniform_belief
                else:
                    new_belief_2 = new_belief_pre_norm_2 / new_belief_pre_norm_sum_2

                calendar.reset(player1_cid = p1_cid + 1, player2_cid = p2_cid + 1)
                env_feedback = calendar.proceed({player1: sampled_real_action})[player1]
                terminate = env_feedback.terminate

                belief_1_1 = new_belief_1
                belief_1_2 = new_belief_2

                if 'msg_success' in env_feedback:
                    msg_succeed.append(env_feedback.msg_success)

                if terminate:
                    if 'action_success' in env_feedback:
                        succeed.append(env_feedback.action_success)
                        action_succeed.append(env_feedback.action_success)
                    else: # 'action_success' in env_feedback
                        succeed.append(False)
                    traj_len.append(t + 1)

            else:
                belief_1_1_stack = np.stack([belief_1_1] * config.num_candidate, axis = 0)
                belief_1_2_stack = np.stack([belief_1_2] * config.num_candidate, axis = 0)
                sampled_action_tea = policy_tea.sample_action(belief_1_1_stack, all_target)
                sampled_action_stu = policy_stu.sample_action(belief_1_2_stack, all_target)

                sampled_real_action = sampled_action_stu[p2_cid]
                belief_update_mask_tea = (sampled_action_tea == sampled_real_action)
                belief_update_mask_stu = (sampled_action_stu == sampled_real_action)
                
                new_belief_pre_norm_1 = belief_update_mask_tea * belief_2_1
                new_belief_pre_norm_sum_1 = np.sum(new_belief_pre_norm_1)
                if new_belief_pre_norm_sum_1 == 0:
                    new_belief_1 = uniform_belief
                else:
                    new_belief_1 = new_belief_pre_norm_1 / new_belief_pre_norm_sum_1
                
                new_belief_pre_norm_2 = belief_update_mask_stu * belief_2_2
                new_belief_pre_norm_sum_2 = np.sum(new_belief_pre_norm_2)
                assert(new_belief_pre_norm_sum_2 != 0)
                new_belief_2 = new_belief_pre_norm_2 / new_belief_pre_norm_sum_2

                calendar.reset(player1_cid = p1_cid + 1, player2_cid = p2_cid + 1)
                env_feedback = calendar.proceed({player2: sampled_real_action})[player2]
                terminate = env_feedback.terminate

                belief_2_1 = new_belief_1
                belief_2_2 = new_belief_2

                if 'msg_success' in env_feedback:
                    msg_succeed.append(env_feedback.msg_success)

                if terminate:
                    if 'action_success' in env_feedback:
                        succ = env_feedback.action_success
                        succeed.append(succ)
                        action_succeed.append(succ)
                    else: # 'action_success' in env_feedback
                        succ = False
                        succeed.append(False)
                    traj_len.append(t + 1)
                    if not succ:
                        p2_wrong_count += 1
            
            t += 1
        
    accuracy = np.mean(succeed)
    wrong_count = len(succeed) - np.sum(succeed)
    p2_wrong = p2_wrong_count / wrong_count
    msg_acc = np.mean(msg_succeed)
    action_acc = np.mean(action_succeed)
    
    print()
    print('[change pair test]: ')
    print('accuracy: ', np.mean(accuracy))
    print('msg_acc: ', np.mean(msg_acc))
    print('action_acc: ', np.mean(action_acc))
    print('p2_wrong: ', np.mean(p2_wrong))
    print('mean_traj_len: ', np.mean(traj_len))
    print()
