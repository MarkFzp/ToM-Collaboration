from easydict import EasyDict as edict
import os

config = edict()

config.gpu = False
config.debug = False
config.verbose = False
config.test = False
config.change_pair_test = False


#########
split = 2
snd_round = False
dataset_path = '/home/luyao/Datasets/Kitchen'
config.ctd_itr = int(17.5e5)
#########

config.ckpt_path = 'test_{}_1/test_aaai_td0.5_entropy0.05_{}_1.ckpt'.format(split, split) if not snd_round \
    else 'test_{}_2/test_aaai_td0.5_entropy0.05_{}_2.ckpt'.format(split, split)
config.ckpt_path_2 = 'test_{}_2/test_aaai_td0.5_entropy0.05_{}_2.ckpt'.format(split, split) if not snd_round \
    else 'test_{}_1/test_aaai_td0.5_entropy0.05_{}_1.ckpt'.format(split, split)
config.load_path = config.ckpt_path
config.dataset_path = os.path.join(dataset_path, 'dataset{}_with_12_ingred_7_dish_5_maxingred_1000000_size_0.7_ratio.npz'.format(split))
config.policy_name_appendix = None if not snd_round else '2'
config.policy_name_appendix_2 = '2'

config.save_itr = 50000
config.print_itr = 100
config.test_itr = 50000
config.itr = 2000000

config.lr = 5e-4  #####
config.batch_size = 100
config.test_batch_size = 10000

config.step_reward = 0  #####
config.fail_reward = -1
config.succeed_reward = 1
config.gamma = 0.6
config.td_lambda = 0.5   #####

config.num_player = 2
config.num_candidate = 7
config.max_attribute = 5
config.num_attribute = 12
config.num_ingredient = config.num_attribute
config.num_action = config.num_attribute

config.p2_target_all_zero = True  #####
config.ad_hoc_structure = False   #####
config.dim_menu_with_context = 3 * config.num_ingredient    #####

config.q_decorrelation = False  #####
config.q_tanh = False  #####
config.q_use_idx = True   #####
config.q_state_dim = config.num_ingredient * config.num_candidate + config.num_candidate + config.num_ingredient
config.fc_layer_dims = [100, 100, config.num_action] #[40, 40, config.num_action]
config.qnet_target_update_itr = 200   #####

config.separate_policy = False  #####
config.hidden_dim = 40
config.in_x_dim = config.num_ingredient * config.num_candidate + config.num_candidate + config.num_ingredient + config.num_action + config.num_player 
config.in_fc_dims = [config.in_x_dim]
config.out_fc_dims = [40, config.num_action]  #####

config.explore_method = 'entropy'
config.entropy_param = 0.05
config.init_epsilon = 0.5   #####
config.end_epsilon = 0.05
config.linear_decay_step = 150000

config.buffer_size = config.batch_size
