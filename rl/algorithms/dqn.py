import copy
from typing import Optional

import torch
from torch.nn import MSELoss

from rl.replay_buffer import ReplayBuffer, TransitionBatch
from rl.environment import RLEnvironment
from rl.policy import RLPolicy


class DQN:
    def __init__(self, estimator, policy: RLPolicy, optimizer, dataset: ReplayBuffer):
        self.estimator = estimator
        self.estimator2 = copy.deepcopy(self.estimator)
        self.policy = policy
        self.agent2 = copy.deepcopy(self.policy)
        self.environment = None
        self.dataset = dataset
        self.optimizer = optimizer

    def _compute_targets(self, transition_batch: TransitionBatch, gamma: float):
        action_dim, transition_dim = 0, 1

        q_value_estimations_next_state: torch.Tensor = \
            self.estimator2(transition_batch.next_state).max(transition_dim).values
        return torch.where(
            transition_batch.done,
            transition_batch.reward,
            transition_batch.reward + gamma * q_value_estimations_next_state
        )

    def train(self, environment: RLEnvironment,
              gamma: float = 0.99, n_episodes: int = 1000, update_rate: int = 50,
              device: torch.device = torch.device('cpu'),
              return_metrics: bool = False, verbose: bool = False):
        self.environment = environment

        self.estimator = self.estimator.to(device)
        self.estimator2 = self.estimator2.to(device)

        if verbose:
            print('Start training')

        if return_metrics:
            all_episodes_rewards = []
            this_episode_rewards = []

        total_step_counter = 0

        for episode_i in range(n_episodes):

            state: torch.Tensor = self.environment.initialize().to(device)
            done = False

            while not done:
                # The agent chooses an action and acts over the environment
                action: torch.Tensor = self.policy.choose_action(self.estimator(state))

                previous_state = state

                state, reward, done, _ = self.environment.act(action)
                state = state.to(device)

                # New transition is stored in the replay buffer
                self.dataset.push(previous_state, action, state, reward, done)

                # Replay buffer is sampled
                transition_batch: Optional[TransitionBatch] = self.dataset.sample()
                if transition_batch is not None:
                    action_dim, transition_dim = 0, 1
                    reward_estimates = self._compute_targets(transition_batch, gamma)

                    # Estimate Q-values of transition states
                    q_value_estimations = self.estimator(transition_batch.state).gather(
                        transition_dim, transition_batch.action.unsqueeze(transition_dim)
                    ).squeeze()

                    # MSE between targets and Q estimations
                    loss = MSELoss()(reward_estimates, q_value_estimations)
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                total_step_counter += 1

                # Clone the estimator every update_rate steps
                if total_step_counter % update_rate == 0:
                    self.estimator2.load_state_dict(self.estimator.state_dict())
                    total_step_counter = 0

                if return_metrics:
                    this_episode_rewards.append(reward)

            if return_metrics:
                all_episodes_rewards.append(torch.sum(torch.tensor(this_episode_rewards)))
                this_episode_rewards = []

            if verbose and (episode_i + 1) % 100 == 0:
                print(f'Episode {episode_i+1}/{n_episodes}. '
                      f'Last 50 eps. reward: {sum(all_episodes_rewards[-50:]) / 50:.2f}')

        if return_metrics:
            return self.estimator, all_episodes_rewards
        else:
            return self.estimator


class DoubleDQN(DQN):
    def _compute_targets(self, transition_batch: TransitionBatch, gamma: float):
        action_dim, transition_dim = 0, 1

        q_value_estimations_next_state: torch.Tensor = \
            self.estimator2(transition_batch.next_state).gather(
                transition_dim,
                self.estimator(transition_batch.next_state)
                .max(transition_dim).indices.unsqueeze(transition_dim)
            ).squeeze()
        return torch.where(
            transition_batch.done,
            transition_batch.reward,
            transition_batch.reward + gamma * q_value_estimations_next_state
        )
