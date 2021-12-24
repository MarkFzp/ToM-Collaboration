

class Config:

    def __init__(self, num_hactions, num_ractions, num_goals):
        self.num_hactions = num_hactions
        self.num_ractions = num_ractions
        self.num_goals = num_goals


class RobotConfig(Config):

    def __init__(self, num_hactions, num_ractions, num_goals):
        super().__init__(num_hactions, num_ractions, num_goals)


class HumanConfig(Config):

    def __init__(self, num_hactions, num_ractions, num_goals):
        super().__init__(num_hactions, num_ractions, num_goals)


class QNetConfig(Config):
    def __init__(self, num_dishes, num_ingredients, embed_dim, num_hactions, num_ractions, num_goals, dense, lr,
                 lr_beta1, lr_epsilon, discount):
        super().__init__(num_hactions, num_ractions, num_goals)
        self.dense = dense
        self.num_dishes = num_dishes
        self.num_ingredients = num_ingredients
        self.embed_dim = embed_dim
        self.lr = lr
        self.lr_beta1 = lr_beta1
        self.lr_epsilon = lr_epsilon
        self.discount = discount
