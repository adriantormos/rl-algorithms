import gym
import torch


class RLEnvironment:
    '''
    Assuming a gym environment
    '''

    def __init__(self, discrete_action_space: bool, state_shape: torch.Size):
        self.state_shape = state_shape
        self.discrete_action_space = discrete_action_space

    def initialize(self):
        pass

    def act(self, action):
        pass

    def is_action_space_discrete(self) -> bool:
        pass


class CartPoleV0Environment(RLEnvironment):
    def __init__(self):
        super(CartPoleV0Environment, self).__init__(True, torch.Size([4]))
        self.environment = gym.make('CartPole-v0')

    def initialize(self):
        state = self.environment.reset()
        return torch.from_numpy(state).float()

    def act(self, action: torch.Tensor):
        state, reward, in_final_state, info = self.environment.step(int(action))
        return torch.from_numpy(state).float(), torch.tensor([reward], dtype=torch.float32), in_final_state, info

    def is_action_space_discrete(self) -> bool:
        return self.discrete_action_space