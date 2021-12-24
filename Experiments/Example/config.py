from easydict import EasyDict as edict

configs = [edict({'batch_size': 128,
                  'ckpt_dir': 'CKPT_soft',
                  'q_learning_rate': 5e-4,
                  'q_learning_gamma': 0.6,
                  'policy_learning_rate': 1e-4,
                  'belief_learning_rate': 1e-3,
                  'split': 1,
                  'memory_size': 1e3,
                  'switch_iters': 5e4}),
           edict({'private_coding_length': 10,
                  'state_coding_length': 10,
                  'action_coding_length': 10,
                  'num_actions': 10,
                  'num_dishes': 4,
                  'max_ingredients': 5,
                  'success_reward': 1,
                  'fail_reward': -1,
                  'step_reward': 0,
                  'acting_boltzman_beta_help': 15,
                  'acting_boltzman_beta_learn': 5,
                  'fixed_fc_layer_info': [20, 10],
                  'fixed_single_length': [20, 10],
                  'fixed_context_length': [8, 6]}),
           edict({'history_dim': 20, 'fc_layer_info': [20, 15, 10]}),
           edict({'history_dim': 20, 'fc_layer_info': [20, 15, 10], 'num_actions': 10}),
           edict({'history_dim': 10, 'fc_layer_info': [20, 15], 'num_actions': 10})]
