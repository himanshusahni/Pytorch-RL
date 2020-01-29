from collections import deque
import numpy as np
import torch
import gym
from skimage import transform


class ExperienceReplay:
    """
    stores tuples of state, action, next state, reward, done
    implemented as a deque
    """
    def __init__(self, size):
        self.buffer = deque(maxlen=size)
        self.size = 0

    def append(self, tuple):
        """
        add a new experience to the buffer
        :param tuple: (s, a, s', r, d)
        """
        self.buffer.append(tuple)
        self.size += 1

    def sample(self, batch_size=1):
        """
        uniformly randomly sample a batch for training
        :param batch_size:
        :return: tuple of (s_batch, a_batch, s'_batch, r_batch, d_batch)
        """
        sample_indices = np.random.randint(0, self.size, size=batch_size)
        samples = [self.buffer[idx] for idx in sample_indices]
        state_batch = torch.cat([torch.unsqueeze(s[0], dim=0) for s in samples], dim=0)
        action_batch = torch.cat([torch.unsqueeze(s[1], dim=0) for s in samples], dim=0)
        next_state_batch = torch.cat([torch.unsqueeze(s[2], dim=0) for s in samples], dim=0)
        reward_batch = torch.cat([s[3] for s in samples], dim=0)
        done_batch = torch.cat([s[4] for s in samples], dim=0)
        return state_batch, action_batch, next_state_batch, reward_batch, done_batch


class ScaledEnv(gym.ActionWrapper):
    """scale aget actions from [-1,1] to action space of environment"""
    def __init__(self, env):
        super().__init__(env)
        self.act_k = (self.action_space.high - self.action_space.low) / 2.
        self.act_k_inv = 2. / (self.action_space.high - self.action_space.low)
        self.act_b = (self.action_space.high + self.action_space.low) / 2.

    def action(self, action):
        return self.act_k * action + self.act_b

    def reverse_action(self, action):
        return self.act_k_inv * (action - self.act_b)


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, reset_freq=100):
        self.reset_freq = reset_freq
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        if self.count % self.reset_freq == 0:
            self.reset()
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class Resize:
    """Resize image input"""
    def __init__(self, size):
        self.size = int(size[0]), int(size[1])

    def __call__(self, img):
        return transform.resize(img, self.size)


class ImgToTensor(object):
    """Convert img to Tensors."""

    def __call__(self, img):
        # swap color axis
        # numpy image: H x W x C
        # torch image: C X H X W
        img = img.transpose((2, 0, 1))
        return torch.from_numpy(img).float()
