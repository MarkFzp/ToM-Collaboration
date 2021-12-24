from easydict import EasyDict as edict

config = edict()

config.gpu = False
config.debug = False
config.verbose = False

config.test = False
config.change_pair_test = False
config.order_free_test = False
assert(sorted([config.test, config.change_pair_test, config.order_free_test]) == [False, False, True] or \
    config.test, config.change_pair_test, config.order_free_test == False, False, False)

#####
split = 3
snd_round = True
config.dataset_path = '/home/luyao/Datasets/Kitchen/dataset{}_with_10_ingred_4_dish_5_maxingred_1000000_size_0.7_ratio.npz'.format(split)
config.continue_itr = 150001
#####

config.load_ckpt = 'test_aaai_entropy0.05_{}_1/test_aaai_entropy0.05_{}_1-150000'.format(split, split) if not snd_round \
    else 'test_aaai_entropy0.05_{}_2/test_aaai_entropy0.05_{}_2-150000'.format(split, split)
config.load_ckpt_2 = 'test_aaai_entropy0.05_{}_2/test_aaai_entropy0.05_{}_2-150000'.format(split, split) if not snd_round \
    else 'test_aaai_entropy0.05_{}_1/test_aaai_entropy0.05_{}_1-150000'.format(split, split)
config.ckpt_path = 'test_aaai_entropy0.05_{}_2/test_aaai_entropy0.05_{}_2'.format(split, split) \
    if snd_round else 'test_aaai_entropy0.05_{}_1/test_aaai_entropy0.05_{}_1'.format(split, split)
config.ckpt_path_2 = None
config.policy_name_appendix = '2' if snd_round else None
config.policy_name_appendix_2 = '2' if not snd_round else None

config.train_one_step = False
config.fixed_menu = False
if config.fixed_menu:
    import numpy as np
    np.random.seed(100)

# config.save_itr = 1000
config.print_itr = 100
config.itr = 1600000
config.test_itr = 10000

config.lr = 2e-4
config.batch_size = 100
config.test_batch_size = 10000

config.step_reward = 0
config.fail_reward = -1
config.succeed_reward = 1
config.gamma = 0.6

config.num_player = 2
config.num_candidate = 4  #7
config.max_attribute = 5  #6
config.num_attribute = 10   #10
config.num_ingredient = config.num_attribute
config.num_action = config.num_attribute

config.reward_architecture = 'adhoc'
config.reward_ckpt_path = 'reward_net_hard'
config.reward_mlp_fc_dim = [100, 100, 100, 100, 1]
config.reward_target_fc_dim = [100, 100, 1]
config.succeed_reward_bias = 20
config.fail_reward_bias = 20
config.reward_itr = 0
config.large_dev_thres = 0.05
config.eps_greedy = [0.85, 0.8235, 0.7857, 0.7273, 0.625, 0.4]

config.strict_imp = True

config.policy_architecture = 'context'
config.policy_fc_dim = [40, 40, config.num_action] #[40, 40, config.num_action]
config.policy_context_dim = [(6, 12)]  #[(6, 12)] [(8, 14)]
config.workplace_fc_dim = 6

config.explore_method = 'entropy'
config.entropy_param = 0.05 #####
config.init_epsilon = 0.5
config.end_epsilon = 0.1
config.linear_decay_step = 2000

config.value_architecture = 'mlp'
config.value_fc_dim = [40, 40, 1] #[40, 40, 1]
