

class Config:

    def __init__(self, num_dishes, num_ingredients, num_ensemble, num_actions, dense, embed_dim, a_lr, c_lr, gamma, batch_size, buffer_size, update_step):
        self.num_dishes = num_dishes
        self.num_ingredients = num_ingredients
        self.num_actions = num_actions
        self.num_ensemble = num_ensemble
        self.dense = dense
        self.embed_dim = embed_dim
        self.a_lr = a_lr
        self.c_lr = c_lr
        self.gamma = gamma
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.update_step = update_step
