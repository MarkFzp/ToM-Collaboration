from abc import ABC, abstractmethod

class Game(ABC):
    def __init__(self, player1, player2, step_reward, fail_reward, succeed_reward):
        self.player1_name_ = player1
        self.player2_name_ = player2
        assert(self.player1_name_ != self.player2_name_)

        self.step_reward_ = step_reward
        self.fail_reward_ = fail_reward
        self.succeed_reward_ = succeed_reward
        self.state_ = None
        super().__init__()
    
    # set up a new game ready to play
    @abstractmethod
    def reset(self):
        pass

    # give initial state to agents, including their private information
    # return: {self.player1_name_: {observation, private},
    #          self.player2_name_: {observation, private}}
    @abstractmethod
    def start(self):
        pass
    
    # proceed the game receiving an action from an agent,
    # state transition function should be implemented within
    # action: {player_name: action}
    # return: {self.player1_name_: (observation, reward},
    #          self.player2_name_: {observation, reward}}
    @abstractmethod
    def proceed(self, action):
        pass
    
    # agents' observation function, can be overload in different games
    def observe(self, player_name):
        assert(self.state_)
        return self.state_