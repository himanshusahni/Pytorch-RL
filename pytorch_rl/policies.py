import torch
import torch.nn.functional as F
from torch.distributions import Categorical, MultivariateNormal

import math
import numpy as np

#TODO: policy now also takes in device
class GaussianPolicy:
    """continuous action policy with fixed Gaussian noise"""

    def __init__(self, device, sigma=0.1, sigma_min=0.1, eps=1., eps_min=0.1, n_steps_annealing=10000):
        """
        :param std: int fixed standard deviation throughout actions
        """
        self.device = device
        self.sigma = sigma
        self.sigma_min = sigma_min
        self.eps = eps
        self.eps_min = eps_min
        self.n_steps = 0
        self.m = -float(sigma - sigma_min) / float(n_steps_annealing)
        self.device = device

    def __call__(self, logits, test=False):
        mean = torch.tanh(logits[0])
        sigma = F.softplus(logits[1]) + 0.01
        action = mean + sigma * torch.randn_like(mean)
        action = torch.clamp(action, -1., 1.)
        self.n_steps += 1
        return action.detach().cpu()

    def entropy(self, logits):
        sigma = F.softplus(logits[1]) + 0.01
        return 0.5 * (1 + (2 * math.pi * sigma.pow(2)).log())

    def logprobs(self, logits, actions):
        mean = torch.tanh(logits[0])
        sigma = F.softplus(logits[1]) + 0.01
        a = - 0.5 * (2 * math.pi * sigma.pow(2)).log()
        b = - ((actions - mean).pow(2)) / (2 * sigma.pow(2))
        return (a + b).sum(dim=1)

    def current_eps(self, test=False):
        if test:
            return 0
        eps = max(self.eps_min , self.m * float(self.n_steps) + self.eps)
        return eps

    def current_sigma(self, test=False):
        if test:
            return self.sigma_min
        sigma = max(self.sigma_min, self.m * float(self.n_steps) + self.sigma)
        return sigma


class OrnsteinUhlenbeckPolicy(GaussianPolicy):
    def __init__(self, theta=0.15, mu=0., sigma=1.0, sigma_min=0.1, dt=1., x0=None, n_steps_annealing=10000):
        super().__init__(sigma, sigma_min, n_steps_annealing)
        self.theta = theta
        self.mu = mu
        self.dt = dt
        self.x_prev = x0

    def __call__(self, logits):
        mean = torch.tanh(logits).detach().cpu().numpy()
        if self.x_prev is None:
            self.x_prev = np.zeros(logits.size())
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + \
            self.current_sigma * np.sqrt(self.dt) * np.random.normal(size=logits.size())
        sampled_action = mean + x
        clipped_action = np.clip(sampled_action, -1, 1)
        self.n_steps += 1
        self.x_prev = x
        return clipped_action


class MultinomialPolicy:

    def __call__(self, logits, test=False):
        """converts logits to probs and samples from multinomial"""
        probs = F.softmax(logits, dim=-1).squeeze()
        if test:
            _, action = torch.max(probs, 0)
        else:
            action = probs.multinomial(num_samples=1).squeeze()
        return action

    def probs(self, logits):
        """just return probabilities"""
        return F.softmax(logits, dim=-1)

    def logprobs(self, logits, actions=None):
        """just return log probabilities"""
        lp = F.log_softmax(logits, dim=-1)
        if actions is None:
            return lp
        else:
            return lp[range(lp.size(0)), actions.flatten()]

    def entropy(self, logits):
        probs = self.probs(logits)
        logprobs = F.log_softmax(logits, dim=-1)
        return (-probs*logprobs).sum(dim=1)


class EpsGreedy:
    """Epsilon-greedy exploration policy for off policy learning"""
    def __init__(self, eps_start, eps_end, eps_steps, action_space):
        self.eps = eps_start
        self.eps_min = eps_end
        self.n_steps = 0
        self.m = -float(eps_start - eps_end) / float(eps_steps)
        self.action_space = action_space

    def current_eps(self):
        return max(self.eps_min , self.m * float(self.n_steps) + self.eps)

    def __call__(self, logits, test=False):
        """converts logits to probs and samples from multinomial"""
        if test or np.random.random() > self.current_eps():
            _, action = torch.max(logits.squeeze(), 0)
        else:
            action = self.action_space.sample()
            action = torch.from_numpy(np.array(action))
        if not test:
            self.n_steps += 1
        return action

    def probs(self, logits):
        """just return probabilities"""
        return F.softmax(logits, dim=-1)

    def logprobs(self, logits, actions=None):
        """just return log probabilities"""
        lp = F.log_softmax(logits, dim=-1)
        if actions is None:
            return lp
        else:
            return lp[range(lp.size(0)), actions.flatten()]

    def entropy(self, logits):
        probs = self.probs(logits)
        logprobs = F.log_softmax(logits, dim=-1)
        return (-probs*logprobs).sum(dim=1)
