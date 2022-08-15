from collections import namedtuple, deque
import random
import torch

TransitionBatch = namedtuple('Transition',
                             ('state', 'action', 'next_state', 'reward', 'done'))


class ReplayBuffer:
    def __init__(self, capacity, batch_size):
        self.dataset = deque([], maxlen=capacity)
        self.batch_size = batch_size

    def push(self, *args):
        """Save a transition"""
        self.dataset.append(TransitionBatch(*args))

    def sample(self):
        if len(self.dataset) < self.batch_size:
            return None
        transition_batch = TransitionBatch(*zip(*random.sample(self.dataset, self.batch_size)))
        transition_batch = TransitionBatch(
            state=torch.stack(transition_batch.state),
            action=torch.tensor(transition_batch.action),
            reward=torch.tensor(transition_batch.reward),
            next_state=torch.stack(transition_batch.next_state),
            done=torch.tensor(transition_batch.done, dtype=torch.bool),
        )
        return transition_batch

    def __len__(self):
        return len(self.dataset)