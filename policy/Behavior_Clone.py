import copy

import os, sys, inspect

import gym
import torch
import numpy as np

# currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
# parentdir = os.path.dirname(currentdir)
# sys.path.insert(0, parentdir)
# print(sys.path)

from ..utils.buffer import OfflineReplayBuffer
from ..net.net import GaussPolicyMLP
from ..value.critic import QLearner
from ..utils.utils import log_prob_func, orthogonal_initWeights

class BehaviorReinforcement:
    _device: torch.device
    _policy: GaussPolicyMLP
    _optimizer: torch.optim
    _policy_lr: float
    _batch_size: int
    def __init__(
            self,
            device: torch.device,
            state_dim: int,
            hidden_dim: int,
            depth: int,
            action_dim: int,
            policy_lr: float,
            batch_size: int,
            rho: float,
    ) -> None:
        super().__init__()
        self._device = device
        self._policy = GaussPolicyMLP(state_dim, hidden_dim, depth, action_dim).to(device)
        orthogonal_initWeights(self._policy)
        self._optimizer = torch.optim.Adam(
            self._policy.parameters(),
            lr=policy_lr
        )
        self._lr = policy_lr
        self._batch_size = batch_size
        self._rho = rho

    # policy loss
    def loss(
            self, replay_buffer: OfflineReplayBuffer, q: QLearner
    ) -> torch.Tensor:
        s, a, _, _, _, _, _, _ = replay_buffer.sample(self._batch_size)
        e_a = self.qfae(s, a, q)
        a_up = a + e_a
        dist = self._policy(s)

        log_prob = log_prob_func(dist, a_up)
        loss = (-log_prob).mean()

        return loss

    # compute QFAE
    def qfae(self, state: torch.Tensor, action: torch.Tensor, q: QLearner):

        a_hat = copy.deepcopy(action)
        a_hat.requires_grad = True
        Q_value = q(state, a_hat).mean()
        a_hat.retain_grad()
        Q_value.backward()
        grad_a = a_hat.grad
        scale = self._rho / (torch.norm(grad_a, p=2, dim=1) + 1e-12)
        e_a = (grad_a * (scale.reshape(-1, 1).repeat(1, a_hat.shape[1]).to(a_hat))).detach()

        return e_a

    # update policy
    def update(
            self, replay_buffer: OfflineReplayBuffer, Q: QLearner
    ) -> float:
        policy_loss = self.loss(replay_buffer, Q)

        self._optimizer.zero_grad()
        policy_loss.backward()
        self._optimizer.step()

        return policy_loss.item()

    # get action
    def select_action(
            self, s: torch.Tensor, is_sample: bool
    ) -> torch.Tensor:
        dist = self._policy(s)
        if is_sample:
            action = dist.sample()
        else:
            action = dist.mean
        return action

    # offline evaluate
    def offline_evaluate(
            self,
            env_name: str,
            seed: int,
            mean: np.ndarray,
            std: np.ndarray,
            eval_episodes: int = 10
    ) -> float:
        env = gym.make(env_name)
        env.seed(seed)

        total_reward = 0
        for _ in range(eval_episodes):
            s, done = env.reset(), False
            while not done:
                s = torch.FloatTensor((np.array(s).reshape(1, -1) - mean) / std).to(self._device)
                a = self.select_action(s, is_sample=False).cpu().data.numpy().flatten()
                s, r, done, _ = env.step(a)
                total_reward += r

        avg_reward = total_reward / eval_episodes
        d4rl_score = env.get_normalized_score(avg_reward) * 100
        return d4rl_score

    # save policy model
    def save(
            self, path: str
    ) -> None:
        torch.save(self._policy.state_dict(), path)
        print('Behavior policy parameters saved in {}'.format(path))

    # load policy model
    def load(
            self, path: str
    ) -> None:
        self._policy.load_state_dict(torch.load(path, map_location=self._device))
        print('Behavior policy parameters loaded')


