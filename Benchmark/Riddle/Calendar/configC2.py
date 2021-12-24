from easydict import EasyDict as edict

config = edict()

config.gpu = False
config.training = True
config.ad_hoc_structure = False
config.log = True
config.epsilon_exp = True
config.soft_max = True
config.print_prob = False
config.dataset = '/home/Datasets/Calendar/calendar_8_1'
config.ckpt_dir = 'CKPT6_1_2'
config.log_file = 'log6_1_2.log'
config.print_iteration = 1000
config.save_frequency = 1000
config.switch_iters = 200000
config.update_q_target_frequency = 100
config.epsilon_min = 0.05
config.epsilon_start = 0.95
config.epsilon_decay = config.switch_iters / 15
config.gamma = 0.6
config.acting_boltzman_beta = 10

config.step_reward = -0.1
config.fail_reward = -1
config.succeed_reward = 1

config.learning_rate = 1e-4
config.batch_size = 50
config.max_batch = 1000
config.iteration = 21000
config.start_iter = 1

config.max_time = config.iteration 
config.num_player = 2
config.num_slot = 8
config.num_obv = config.num_slot + config.num_slot
config.num_candidate = config.num_calender = 2 ** config.num_slot - 1
config.num_action = int((1 + config.num_slot) * config.num_slot / 2 + config.num_slot + 1)
config.num_fingerprint = 2

config.layer_dim = [50, 50, config.num_action]
config.out_fc_dims = [50, config.num_action]

config.in_x_dim = config.num_obv + config.num_action
config.out_x_dim = config.in_x_dim
# config.hidden_dim = config.num_action
config.hidden_dim = config.out_x_dim


