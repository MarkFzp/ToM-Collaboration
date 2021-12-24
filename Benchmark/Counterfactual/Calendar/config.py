from easydict import EasyDict as edict
import os

config = edict()

config.gpu = False
config.debug = False
config.verbose = False

config.test = False
config.change_pair_test = True
config.max_len_msg_test = False

### teamviewer ###
split = 3
snd_round = True
dataset_calendar_folder = '/home/zipeng/Datasets/Calendar'
config.gpu_name = 'gpu:0'
###

config.load_ckpt = 'aaai_{}_1/aaai_{}_1.ckpt-120000'.format(split, split) if not snd_round else 'aaai_{}_2/aaai_{}_2.ckpt-120000'.format(split, split)
config.load_ckpt_2 = 'aaai_{}_2/aaai_{}_2.ckpt-120000'.format(split, split) if not snd_round else 'aaai_{}_1/aaai_{}_1.ckpt-120000'.format(split, split)
config.ckpt_path = 'aaai_{}_1/aaai_{}_1.ckpt'.format(split, split) if not snd_round else 'aaai_{}_2/aaai_{}_2.ckpt'.format(split, split)
config.ckpt_path_2 = None
config.policy_name_appendix = None if not snd_round else '2'
config.policy_name_appendix_2 = '2' if not snd_round else None
config.dataset_path_prefix = os.path.join(dataset_calendar_folder, 'calendar_8_{}'.format(split))

config.print_itr = 100
config.test_itr = 1000
config.itr = 200000

config.lr = 5e-4  #####
config.batch_size = 100
config.test_batch_size = 10000

config.step_reward = -0.1  #####
config.fail_reward = -2
config.succeed_reward = 1
config.gamma = 0.6
config.td_lambda = 0.5   #####

config.num_player = 2
config.num_slot = 8
config.num_candidate = config.num_calendar = 2 ** config.num_slot # 255
config.num_action = int((1 + config.num_slot) * config.num_slot / 2 + config.num_slot + 1) # 45
config.action_encode_dim = 16
config.terminate_len = 8

config.qnet_hidden_dim = 30
config.fc_layer_dims = [46, 46, config.num_action]  #[100, 100, config.num_action]
config.qnet_target_update_itr = 500   #####

config.policy_hidden_dim = 40
config.in_fc_dims = [50]
config.out_fc_dims = [40, 40, config.num_action]  #####
config.in_x_dim = config.num_slot + config.action_encode_dim + 2 + 2

config.entropy_param = 0.2
