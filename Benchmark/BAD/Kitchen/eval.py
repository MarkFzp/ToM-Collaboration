import numpy as np
from copy import deepcopy
from collections import Counter
from config import config

p2_target = np.zeros([config.num_candidate])
p2_target_2d = np.expand_dims(p2_target, axis = 0)
all_target = np.eye(config.num_candidate)
uniform_belief = np.ones(config.num_candidate) / config.num_candidate
uniform_belief_tile = np.stack([uniform_belief] * config.num_candidate, axis = 0)
player1 = 0 # teacher
player2 = 1 # student

def change_pair_test(kitchen, policy_tea, policy_stu, reward_net):

    succeed = []
    traj_len = []
    action_diff_freq = Counter()
    unif_b_after_1_count = 0
    p2_wrong_count = 0
    total_epoch_count = 0

    for _ in range(config.test_batch_size):
        kitchen.reset(train = False)
        start_ret = kitchen.start()
        p1_start_ret = start_ret[player1]

        menu = p1_start_ret.observation.menu_embed
        menu_tile = np.stack([menu] * config.num_candidate, axis = 0)
        menu_3d = np.expand_dims(menu, axis = 0)

        workplace = deepcopy(p1_start_ret.observation.workplace_embed)

        p1_target = p1_start_ret.private
        p1_target_idx = np.where(p1_target)[0][0]

        belief = uniform_belief

        t = 0
        terminate = False
        while not terminate:
            if config.train_one_step and t > 0:
                succeed.append(True)
                traj_len.append(t + 1)
                break

            if t % 2 == 0:
                workplace_tile = np.stack([workplace] * config.num_candidate, axis = 0)
                sampled_action_tea, prob_pre_anneal_tea = policy_tea.sample_action(menu_tile, workplace_tile, all_target)
                sampled_action_stu, prob_pre_anneal_stu = policy_stu.sample_action(menu_tile, workplace_tile, all_target)

                if t == 0:
                    action_diff_freq[np.sum(sampled_action_tea == sampled_action_stu)] += 1

                sampled_real_action_tea = sampled_action_tea[p1_target_idx]
                belief_update_mask = (sampled_action_stu == sampled_real_action_tea)
                new_belief_pre_norm = belief_update_mask * belief
                new_belief_pre_norm_sum = np.sum(new_belief_pre_norm)
                
                if new_belief_pre_norm_sum == 0:
                    new_belief = uniform_belief
                    if t == 0:
                        unif_b_after_1_count += 1
                else:
                    new_belief = new_belief_pre_norm / new_belief_pre_norm_sum

                env_feedback = kitchen.proceed({player1: sampled_real_action_tea})[player1]
                terminate = env_feedback.terminate
                new_workplace = deepcopy(env_feedback.observation.workplace_embed)

                belief = new_belief
                workplace = new_workplace

                if terminate:
                    succeed.append(env_feedback.success)
                    traj_len.append(t + 1)

            else:
                workplace_2d = np.expand_dims(workplace, axis = 0)

                sampled_action, prob_pre_anneal = policy_stu.sample_action(
                    menu_3d, 
                    workplace_2d, 
                    np.expand_dims(belief, axis = 0)
                )

                sampled_action = sampled_action[0]
                
                env_feedback = kitchen.proceed({player2: sampled_action})[player2]
                terminate = env_feedback.terminate
                new_workplace = deepcopy(env_feedback.observation.workplace_embed)

                workplace = new_workplace

                if terminate:
                    succ = env_feedback.success
                    succeed.append(succ)
                    traj_len.append(t + 1)
                    if not succ:
                        p2_wrong_count += 1
            
            total_epoch_count += 1
            t += 1
    
    accuracy = np.mean(succeed)
    wrong_count = len(succeed) - np.sum(succeed)
    p2_wrong = p2_wrong_count / wrong_count
    
    
    print('accuracy: ', accuracy)
    print('traj_len: ', np.mean(traj_len))
    print('p2_wrong: ', p2_wrong)
    print('first_stu_action_diff_freq: ', ['{}: {:.2%}'.format(i, action_diff_freq[i] / config.test_batch_size) for i in action_diff_freq])
    print('unif_b_after_1_action: {:.4%}'.format(unif_b_after_1_count / config.test_batch_size))


def unique_id(config, menu, target_id, action_0):
    other_context = np.sum(np.delete(menu, target_id, axis = 0), axis = 0)
    target_one_hot = menu[target_id]
    target_ingred = np.where(target_one_hot)[0]
    target_ingred_is_unique_id = np.logical_not(other_context[target_ingred])
    unique_id_exist = True in target_ingred_is_unique_id
    if not unique_id_exist:
        return False, False
    elif action_0 in target_ingred and target_ingred_is_unique_id[np.where(target_ingred == action_0)[0][0]]:
        return True, True
    else:
        return True, False


def test(kitchen, policy_net, reward_net):

    succeed = []
    traj_len = []
    has_unique_id_l = []
    sent_unique_id_l = []
    p2_wrong_count = 0
    updated_belief_len = Counter()
    b_len_l = []

    for _ in range(config.test_batch_size):
        kitchen.reset(train = False)
        start_ret = kitchen.start()
        p1_start_ret = start_ret[player1]

        menu = p1_start_ret.observation.menu_embed
        menu_tile = np.stack([menu] * config.num_candidate, axis = 0)
        menu_3d = np.expand_dims(menu, axis = 0)

        workplace = deepcopy(p1_start_ret.observation.workplace_embed)

        p1_target = p1_start_ret.private
        # p1_target_tile = np.stack([p1_target] * config.num_candidate, axis = 0)
        # p1_target_2d = np.expand_dims(p1_target, axis = 0)
        p1_target_idx = np.where(p1_target)[0][0]

        belief = uniform_belief

        t = 0
        terminate = False
        while not terminate:
            if config.train_one_step and t > 0:
                succeed.append(True)
                traj_len.append(t + 1)
                break

            if t % 2 == 0:
                workplace_tile = np.stack([workplace] * config.num_candidate, axis = 0)
                sampled_action, prob_pre_anneal = policy_net.sample_action(menu_tile, workplace_tile, all_target)

                sampled_real_action = sampled_action[p1_target_idx]
                belief_update_mask = (sampled_action == sampled_real_action)
                new_belief_pre_norm = belief_update_mask * belief
                new_belief_pre_norm_sum = np.sum(new_belief_pre_norm)
                assert(new_belief_pre_norm_sum != 0)
                new_belief = new_belief_pre_norm / new_belief_pre_norm_sum

                if t == 0:
                    b_len = np.where(new_belief)[0].size
                    updated_belief_len[b_len] += 1
                    b_len_l.append(b_len)
                    has_unique_id, sent_unique_id = unique_id(config, menu, p1_target_idx, sampled_real_action)
                    has_unique_id_l.append(has_unique_id)
                    sent_unique_id_l.append(sent_unique_id)

                env_feedback = kitchen.proceed({player1: sampled_real_action})[player1]
                terminate = env_feedback.terminate
                new_workplace = deepcopy(env_feedback.observation.workplace_embed)

                belief = new_belief
                workplace = new_workplace

                if terminate:
                    succeed.append(env_feedback.success)
                    traj_len.append(t + 1)

            else:
                workplace_2d = np.expand_dims(workplace, axis = 0)

                sampled_action, prob_pre_anneal = policy_net.sample_action(
                    menu_3d, 
                    workplace_2d, 
                    np.expand_dims(belief, axis = 0)
                )

                sampled_action = sampled_action[0]
                prob_pre_anneal = prob_pre_anneal[0]
                # bad_reward = np.sum(belief * reward)
                
                env_feedback = kitchen.proceed({player2: sampled_action})[player2]
                terminate = env_feedback.terminate
                new_workplace = deepcopy(env_feedback.observation.workplace_embed)

                workplace = new_workplace

                if terminate:
                    succ = env_feedback.success
                    succeed.append(succ)
                    traj_len.append(t + 1)
                    if not succ:
                        p2_wrong_count += 1
            
            t += 1

    
    assert(len(succeed) == len(traj_len) == len(sent_unique_id_l) == len(has_unique_id_l) == len(b_len_l) == config.test_batch_size)

    succeed = np.array(succeed)
    accuracy = np.mean(succeed)
    wrong_count = len(succeed) - np.sum(succeed)
    p2_wrong = p2_wrong_count / wrong_count

    sent_unique_id_l = np.array(sent_unique_id_l)
    has_unique_id_l = np.array(has_unique_id_l)
    assert(np.mean(sent_unique_id_l * has_unique_id_l == sent_unique_id_l) == 1)
    has_unique_id_rate = np.mean(has_unique_id_l)
    sent_unique_id_rate = np.mean(sent_unique_id_l[np.where(has_unique_id_l)[0]])
    sent_unique_id_succ = np.mean(succeed[np.where(sent_unique_id_l)[0]])
    not_sent_unique_id_succ = np.mean(succeed[np.where(has_unique_id_l * (1 - sent_unique_id_l))[0]])

    print('acc: ', accuracy)
    print('traj_len: ', np.mean(traj_len))
    print('p2_wrong: ', p2_wrong)
    print('has_unique_id_rate: ', has_unique_id_rate)
    print('sent_unique_id_rate: ', sent_unique_id_rate)
    print('sent_unique_id_succ: ', sent_unique_id_succ)
    print('not_sent_unique_id_succ: ', not_sent_unique_id_succ)
    
    b_len_l = np.array(b_len_l)
    for length in range(1, config.num_candidate):
        b_len_l_idx = np.where(b_len_l == length)[0]
        if b_len_l_idx.size != 0:
            rate = np.mean(succeed[b_len_l_idx])
            print('acc of {} belief: '.format(length), rate)
    
    print('updated_belief_len: ', ['{}: {:.2%}'.format(i, updated_belief_len[i] / config.test_batch_size) for i in updated_belief_len])


def order_free_test(kitchen, policy_net, reward_net):
    succeed = []
    traj_len = []
    has_unique_id_l = []
    sent_unique_id_l = []
    p2_wrong_count = 0
    updated_belief_len = Counter()
    b_len_l = []

    row_perm = np.random.permutation(config.num_candidate)
    col_perm = np.random.permutation(config.num_ingredient)
    print('row_perm: ', row_perm)
    print('col_perm: ', col_perm)

    for _ in range(config.test_batch_size):
        kitchen.reset(train = False)
        start_ret = kitchen.start()
        p1_start_ret = start_ret[player1]

        menu = p1_start_ret.observation.menu_embed
        menu_stu = menu[row_perm][:, col_perm]
        menu_tile = np.stack([menu] * config.num_candidate, axis = 0)
        # menu_3d = np.expand_dims(menu, axis = 0)
        menu_3d_stu = np.expand_dims(menu_stu, axis = 0)

        workplace = deepcopy(p1_start_ret.observation.workplace_embed)

        p1_target = p1_start_ret.private
        # p1_target_tile = np.stack([p1_target] * config.num_candidate, axis = 0)
        # p1_target_2d = np.expand_dims(p1_target, axis = 0)
        p1_target_idx = np.where(p1_target)[0][0]

        belief = uniform_belief

        t = 0
        terminate = False
        while not terminate:
            if config.train_one_step and t > 0:
                succeed.append(True)
                traj_len.append(t + 1)
                break

            if t % 2 == 0:
                workplace_tile = np.stack([workplace] * config.num_candidate, axis = 0)
                sampled_action, prob_pre_anneal = policy_net.sample_action(menu_tile, workplace_tile, all_target)

                sampled_real_action = sampled_action[p1_target_idx]
                belief_update_mask = (sampled_action == sampled_real_action)
                new_belief_pre_norm = belief_update_mask * belief
                new_belief_pre_norm_sum = np.sum(new_belief_pre_norm)
                assert(new_belief_pre_norm_sum != 0)
                new_belief = new_belief_pre_norm / new_belief_pre_norm_sum

                if t == 0:
                    b_len = np.where(new_belief)[0].size
                    updated_belief_len[b_len] += 1
                    b_len_l.append(b_len)
                    has_unique_id, sent_unique_id = unique_id(config, menu, p1_target_idx, sampled_real_action)
                    has_unique_id_l.append(has_unique_id)
                    sent_unique_id_l.append(sent_unique_id)

                env_feedback = kitchen.proceed({player1: sampled_real_action})[player1]
                terminate = env_feedback.terminate
                new_workplace = deepcopy(env_feedback.observation.workplace_embed)

                belief = new_belief
                workplace = new_workplace

                if terminate:
                    succeed.append(env_feedback.success)
                    traj_len.append(t + 1)

            else:
                workplace_2d_stu = np.expand_dims(workplace[col_perm], axis = 0)
                
                sampled_action_stu, prob_pre_anneal = policy_net.sample_action(
                    menu_3d_stu, 
                    workplace_2d_stu, 
                    np.expand_dims(belief[row_perm], axis = 0)
                )

                sampled_action_stu = sampled_action_stu[0]
                sampled_action = col_perm[sampled_action_stu]
                
                
                env_feedback = kitchen.proceed({player2: sampled_action})[player2]
                terminate = env_feedback.terminate
                new_workplace = deepcopy(env_feedback.observation.workplace_embed)

                workplace = new_workplace

                if terminate:
                    succ = env_feedback.success
                    succeed.append(succ)
                    traj_len.append(t + 1)
                    if not succ:
                        p2_wrong_count += 1
            
            t += 1

    assert(len(succeed) == len(traj_len) == len(sent_unique_id_l) == len(has_unique_id_l) == len(b_len_l) == config.test_batch_size)

    succeed = np.array(succeed)
    accuracy = np.mean(succeed)
    wrong_count = len(succeed) - np.sum(succeed)
    p2_wrong = p2_wrong_count / wrong_count

    sent_unique_id_l = np.array(sent_unique_id_l)
    has_unique_id_l = np.array(has_unique_id_l)
    assert(np.mean(sent_unique_id_l * has_unique_id_l == sent_unique_id_l) == 1)
    has_unique_id_rate = np.mean(has_unique_id_l)
    sent_unique_id_rate = np.mean(sent_unique_id_l[np.where(has_unique_id_l)[0]])
    sent_unique_id_succ = np.mean(succeed[np.where(sent_unique_id_l)[0]])
    not_sent_unique_id_succ = np.mean(succeed[np.where(has_unique_id * (1 - sent_unique_id_l))[0]])

    print('acc: ', accuracy)
    print('traj_len: ', np.mean(traj_len))
    print('p2_wrong: ', p2_wrong)
    print('has_unique_id_rate: ', has_unique_id_rate)
    print('sent_unique_id_rate: ', sent_unique_id_rate)
    print('sent_unique_id_succ: ', sent_unique_id_succ)
    print('not_sent_unique_id_succ: ', not_sent_unique_id_succ)
    
    b_len_l = np.array(b_len_l)
    for length in range(1, config.num_candidate):
        b_len_l_idx = np.where(b_len_l == length)[0]
        if b_len_l_idx.size != 0:
            rate = np.mean(succeed[b_len_l_idx])
            print('acc of {} belief: '.format(length), rate)
    
    print('updated_belief_len: ', ['{}: {:.2%}'.format(i, updated_belief_len[i] / config.test_batch_size) for i in updated_belief_len])

