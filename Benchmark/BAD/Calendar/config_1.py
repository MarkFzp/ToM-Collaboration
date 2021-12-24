from easydict import EasyDict as edict
import os

config = edict()

config.gpu = True
config.debug = False
config.verbose = False

config.test = False
config.change_pair_test = False

### teamviewer ###
split = 1
snd_round = False
dataset_calendar_folder = '/home/luyao/Datasets/Calendar'
config.gpu_name = 'gpu:1'
config.ctd_itr = 95001
###

config.load_ckpt = 'aaai_{}_1/aaai_{}_1.ckpt-95000'.format(split, split) if not snd_round else 'aaai_{}_2/aaai_{}_2.ckpt-95000'.format(split, split)
config.load_ckpt_2 = 'aaai_{}_1/aaai_{}_1.ckpt-95000'.format(split, split) if snd_round else 'aaai_{}_2/aaai_{}_2.ckpt-95000'.format(split, split)
config.ckpt_path = 'aaai_{}_1/aaai_{}_1.ckpt'.format(split, split) if not snd_round else 'aaai_{}_2/aaai_{}_2.ckpt'.format(split, split)
config.ckpt_path_2 = 'aaai_{}_1/aaai_{}_1.ckpt'.format(split, split) if snd_round else 'aaai_{}_2/aaai_{}_2.ckpt'.format(split, split)
config.policy_name_appendix = None if not snd_round else '2'
config.policy_name_appendix_2 = None if snd_round else '2'
config.dataset_path_prefix = os.path.join(dataset_calendar_folder, 'calendar_8_{}'.format(split))

# config.save_itr = 1000
config.print_itr = 100
config.itr = 1600000
config.test_itr = 10000

config.lr = 5e-4
config.value_lr_increase_factor = 1
config.batch_size = 64
config.test_batch_size = 10000

config.step_reward = -0.1
config.fail_reward = -2
config.succeed_reward = 1
config.gamma = 0.6
config.terminate_len = 8

config.num_player = 2
config.num_slot = 8
config.num_candidate = config.num_calendar = 2 ** config.num_slot - 1
config.num_action = (1 + config.num_slot) * config.num_slot / 2 + config.num_slot + 1

config.policy_architecture = 'context'
config.context_matmul = False
config.policy_context_dim = [8]
# config.policy_fc_dim = [32, 32, config.num_action] #
config.policy_fc_dim = [66, 66, config.num_action]
# config.cf_fc_dim = [51, 8] #
config.cf_fc_dim = [255, 25]

config.explore_method = 'entropy'
config.entropy_param = 0.2 #####
config.init_epsilon = 0.5
config.end_epsilon = 0.1
config.linear_decay_step = 2000

config.value_architecture = 'mlp2'
config.value_fc_dim = [255, 100, 1] if config.value_architecture == 'mlp' else [80, 40, 1]
