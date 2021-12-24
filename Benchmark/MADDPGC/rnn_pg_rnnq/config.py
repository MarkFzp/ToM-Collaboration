class Config:

    def __init__(self, num_slots, num_ensemble, num_actions, dense, embed_dim, a_lr, c_lr, init_global_step, lr_decay_step,
                 lr_decay_rate, gamma, lamb, beta, batch_size, buffer_size, max_traj_len,update_step, break_correlation, use_gpu):
        self.num_slots = num_slots
        self.num_actions = num_actions
        self.num_ensemble = num_ensemble
        self.dense = dense
        self.embed_dim = embed_dim
        self.a_lr = a_lr
        self.c_lr = c_lr
        self.init_global_step = init_global_step
        self.lr_decay_step = lr_decay_step
        self.lr_decay_rate = lr_decay_rate
        self.gamma = gamma
        self.lamb = lamb
        self.beta = beta
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.max_traj_len = max_traj_len
        self.update_step = update_step
        self.break_correlation = break_correlation
        self.use_gpu = use_gpu