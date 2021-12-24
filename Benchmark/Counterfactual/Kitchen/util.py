import numpy as np
from copy import deepcopy

def change_pair_test(config, player1, player2, kitchen, policy1, policy2, buffer, count):
    global null_state
    null_state = np.zeros([config.q_state_dim], dtype = np.int32)
    global null_action_one_hot
    null_action_one_hot = np.zeros([config.num_action], dtype = np.int32)
    global all_action
    all_action = np.arange(config.num_action, dtype = np.int32)
    global p2_target
    if config.ad_hoc_structure:
        p2_target = np.ones([config.num_candidate]) / config.num_candidate
    elif config.p2_target_all_zero:
        p2_target = np.zeros([config.num_candidate], dtype = np.int32)
    else:
        p2_target = np.ones([config.num_candidate]) / config.num_candidate

    global one_hot_action
    one_hot_action = np.eye(config.num_action)
    global one_hot_p_idx
    one_hot_p_idx = np.eye(config.num_player)

    acc_list = []

    for _ in range(count):
        kitchen.reset(train = False)
        start_ret = kitchen.start()
        p1_start_ret = start_ret[player1]
        p2_start_ret = start_ret[player2]
        menu = p1_start_ret['observation'].menu_embed
        menu_len = len(menu)
        # if null_action_one_hot in menu:
        #     print(menu)
        #     raise Exception('null dish in menu')
        # zero padding to menu
        if menu_len != config.num_candidate:
            print('found short menu!!!!!')
            assert(menu_len < config.num_candidate)
            menu = np.stack([menu, np.zeros([config.num_candidate - menu_len, config.num_ingredient])], axis = 0)
        menu = menu.reshape([-1])

        p1_target = deepcopy(p1_start_ret['private'])
        p1_menu_and_target = np.concatenate([menu, p1_target])
        p2_menu_and_target = np.concatenate([menu, p2_target])


        # player 1 acts first
        # player 1 & 2 act turn by turn
        terminate = False
        t = 0
        in_state = [None, None]
        prev_action = [-1, -1]
        prepared_ingred = np.zeros([config.num_ingredient])
        
        trajectory = {'menu': menu, 'p1_target': p1_target, 'sarsa': [], 'success': None}
        
        while not terminate:
            p1_obs = np.concatenate([p1_menu_and_target, prepared_ingred])
            p1_in_action = prev_action[0]
            p1_in_state = in_state[0]
            p1_out_action, p1_out_action_prob, p1_out_state = policy1.single_sample_act(p1_obs, p1_in_action, player1, p1_in_state)
            p2_obs = np.concatenate([p2_menu_and_target, prepared_ingred])
            p2_in_action = prev_action[1]
            p2_in_state = in_state[1]
            p2_out_action, p2_out_action_prob, p2_out_state = policy2.single_sample_act(p2_obs, p2_in_action, player2, p2_in_state)
            # pdb.set_trace()
            if t % 2 == 0:
                env_feedback = kitchen.proceed({player1: p1_out_action})[player1]
                new_prepared_ingred = env_feedback.observation.workplace_embed
                reward = env_feedback.reward
                terminate = env_feedback.terminate
                
                if terminate:
                    trajectory['success'] = env_feedback.success
                    acc_list.append(env_feedback.success)
                trajectory['sarsa'].append((prepared_ingred, p1_out_action, reward, p1_out_action_prob))
                
                in_state = [p1_out_state, p2_out_state]
                prev_action = [p1_out_action, -1]
                prepared_ingred = deepcopy(new_prepared_ingred)
            else:
                env_feedback = kitchen.proceed({player2: p2_out_action})[player2]
                new_prepared_ingred = env_feedback.observation.workplace_embed
                reward = env_feedback.reward
                terminate = env_feedback.terminate

                if terminate:
                    trajectory['success'] = env_feedback.success
                    acc_list.append(env_feedback.success)
                trajectory['sarsa'].append((prepared_ingred, p2_out_action, reward, p2_out_action_prob))
                
                in_state = [p1_out_state, p2_out_state]
                prev_action = [-1, p2_out_action]
                prepared_ingred = deepcopy(new_prepared_ingred)

            t += 1

    print('[test acc]: ', np.mean(acc_list))

