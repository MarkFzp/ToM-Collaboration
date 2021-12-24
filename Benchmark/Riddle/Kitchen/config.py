from easydict import EasyDict as edict

config = edict()

config.dataset = '/home/Datasets/Kitchen/dataset1_with_10_ingred_4_dish_5_maxingred_1000000_size_0.7_ratio.npz'
config.gpu = False
config.training = True
config.ad_hoc_structure = False
config.log = True
config.epsilon_exp = True
config.soft_max = False
config.print_prob = False
config.exp_folder = './'
config.ckpt_path = 'CKPT'
config.log_dir = './'
config.log_file = 'log_test4.log'
config.start_iter = int(3e6)
config.iteration = int(4e6)

config.print_iteration = 1000
config.save_frequency = 10000
config.switch_iters = 200000
config.update_q_target_frequency = 100
config.epsilon_min = 0
config.epsilon_start = 0
config.epsilon_decay = config.switch_iters / 10
config.gamma = 0.6
config.acting_boltzman_beta = 15
config.fail_p = 0.6

config.step_reward = 0
config.fail_reward = -1
config.succeed_reward = 1

config.learning_rate = 1e-4
config.batch_size = 50
config.max_batch = 1000

config.max_time = config.iteration 
config.num_player = 2
config.num_candidate = 4
config.max_attribute = 5
config.num_attribute = 10
config.num_ingredient = config.num_attribute
config.num_action = config.num_attribute
config.num_obv = config.num_candidate + 2
config.num_fingerprint = 2

config.layer_dim = [50, 50, config.num_action]
config.out_fc_dims = [50, config.num_action]
config.dim_menu_with_context = 4 * config.num_ingredient

config.in_x_dim = config.num_obv * config.num_ingredient + config.num_action
config.out_x_dim = config.in_x_dim
# config.hidden_dim = config.num_action
config.hidden_dim = config.out_x_dim


