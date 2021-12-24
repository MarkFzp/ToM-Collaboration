from easydict import EasyDict as edict
import os

config = edict()

config.gpu = False
config.debug = False
config.verbose = False

config.test = False
config.change_pair_test = False
config.max_len_msg_test = False

### teamviewer ###
split = 3
snd_round = True
dataset_calendar_folder = '/home/Datasets/Calendar'
config.gpu_name = 'gpu:0'
large = True
deep = False
entropy = 0.2
td = 0.5
ctd_itr = 920001
###

config.load_ckpt = 'aaai_{}_1/aaai_{}_1.ckpt-920000'.format(split, split) if not snd_round else 'aaai_{}_2/aaai_{}_2.ckpt-920000'.format(split, split)
config.load_ckpt_2 = 'aaai_{}_2/aaai_{}_2.ckpt-920000'.format(split, split) if not snd_round else 'aaai_{}_1/aaai_{}_1.ckpt-920000'.format(split, split)
config.ckpt_path = 'aaai_{}_1/aaai_{}_1.ckpt'.format(split, split) if not snd_round else 'aaai_{}_2/aaai_{}_2.ckpt'.format(split, split)
config.ckpt_path_2 = None
config.policy_name_appendix = None if not snd_round else '2'
config.policy_name_appendix_2 = '2' if not snd_round else None
config.dataset_path_prefix = os.path.join(dataset_calendar_folder, 'calendar_8_{}'.format(split))

config.print_itr = 100
config.test_itr = 10000
config.itr = 1600000
config.ctd_itr = ctd_itr

config.lr = 5e-4  #####
config.batch_size = 100
config.test_batch_size = 10000

config.step_reward = -0.1  #####
config.fail_reward = -2
config.succeed_reward = 1
config.gamma = 0.6
config.td_lambda = td   #####

config.num_player = 2
config.num_slot = 8
config.num_candidate = config.num_calendar = 2 ** config.num_slot # 255
config.num_action = int((1 + config.num_slot) * config.num_slot / 2 + config.num_slot + 1) # 45
config.action_encode_dim = 16
config.terminate_len = 8

config.fc_layer_dims = ([60, 60, 60, config.num_action] if deep else [60, 60, config.num_action]) if large \
    else [34, 34, config.num_action] #[100, 100, config.num_action]
config.qnet_target_update_itr = 500   #####

config.policy_hidden_dim = 45 if large else 28
config.in_fc_dims = [50] if large else [28]
config.out_fc_dims = ([60, 60, 60, config.num_action] if deep else [60, 60, config.num_action]) if large \
    else [28, 28, config.num_action]#####
config.in_x_dim = config.num_slot + config.action_encode_dim + 2 + 2  # 28

config.entropy_param = entropy
