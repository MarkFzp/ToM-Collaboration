from easydict import EasyDict as edict 
import os
config = edict()

config.gpu = False
config.debug = False
config.verbose = False
config.fixed_menu = False

config.test = False
config.change_pair_test = False
config.constant_message_test = False
assert(sorted([config.test, config.change_pair_test, config.constant_message_test]) in [[False, False, True], [False, False, False]])

#########
split = 3
snd_round = False
dataset_path = '/home/luyao/Datasets/Kitchen'
#########

config.ckpt_path = 'ckpt/test_{}_1.ckpt'.format(split) if not snd_round \
    else 'ckpt/test_{}_2.ckpt'.format(split)
config.ckpt_path_2 = 'ckpt/test_{}_2.ckpt'.format(split) if not snd_round \
    else 'ckpt/test_{}_1.ckpt'.format(split)
config.dataset_path = os.path.join(dataset_path, 'dataset{}_with_12_ingred_7_dish_5_maxingred_1000000_size_0.7_ratio.npz'.format(split))
config.qnet_name_appendix = None if not snd_round else '2'
config.qnet_name_appendix_2 = '2' if not snd_round else None
config.qnet_target_update_itr = 500

# config.save_itr = 1000
config.print_itr = 100
config.itr = 400000
config.test_itr = 1000

config.lr = 2e-4
config.batch_size = 100
config.test_batch_size = 10000

config.step_reward = 0
config.fail_reward = -1
config.succeed_reward = 1
config.gamma = 0.6

config.num_player = 2
config.num_candidate = 7  #4
config.max_attribute = 5  #5
config.num_attribute = 12   #10
config.num_ingredient = config.num_attribute
config.num_action = config.num_attribute

config.init_eps = 0.9
config.end_eps = 0.05
config.linear_decay_step = 50000

config.message_activation = 'sigmoid'
config.message_dim = 7 #

config.in_arch = 'mlp'
config.out_arch = 'mlp'
config.in_fc_dims = [128]
config.out_fc_dims = [128, config.num_action + config.message_dim]
config.hidden_dims = [128, 128]
config.noise_std = 1   #
