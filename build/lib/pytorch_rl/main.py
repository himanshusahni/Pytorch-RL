import argparse
import gym
import torch
import torchvision.transforms as transforms

from pytorch_rl import networks, utils, callbacks, agents, algorithms, policies

if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--replay_warmup',
                        help='states in the replay before training starts',
                        type=int,
                        default=1000)
    parser.add_argument('--replay_capacity',
                        help='max length of replay buffer',
                        type=int,
                        default=1000000)
    parser.add_argument('--max_train_steps',
                        help='maximum environment steps allowed for training',
                        type=int,
                        default=1000000)
    parser.add_argument('--batch_size',
                        help='batch size for training policy and value networks',
                        type=int,
                        default=128)
    parser.add_argument('--obs_size',
                        help='resize observations from environment',
                        type=int,
                        default=64)
    parser.add_argument('--gamma',
                        help='discount factor',
                        type=float,
                        default=0.99)
    parser.add_argument('--nb_threads',
                        help='number of processes for multithreaded algorithms',
                        type=int,
                        default=10)
    parser.add_argument('--nb_rollout_steps',
                        help='steps per rollout for AC type algorithms',
                        type=int,
                        default=10)
    parser.add_argument('--test_freq',
                        help='how often to test',
                        type=int,
                        default=1000)
    parser.add_argument('--env',
                        help='environment name',)
    args = parser.parse_args()

    # args.env = 'Breakout-v4'
    args.env = 'LunarLander-v2'
    def make_env():
        return gym.make(args.env)
    env = make_env()

    # args.observation_space = (3, args.obs_size, args.obs_size)
    args.observation_space = env.observation_space.shape
    args.action_space = env.action_space.n
    args.device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

    # policy_network = networks.ConvPolicy_64x64(args.action_space)
    # value_network = networks.ConvValue_64x64()
    policy_network = networks.FCPolicy(args.observation_space[0], args.action_space)
    value_network = networks.FCValue(args.observation_space[0])
    # ddpg = algorithms.TD3(policy_network, value_network, args)
    # policy = policies.OrnsteinUhlenbeckPolicy(env.action_space.low.shape)
    policy = policies.MultinomialPolicy()
    # plotter = VisdomLinePlotter(env_name='ddpg')
    # callbacks = [callbacks.PlotCallback(plotter, freq=100), callbacks.PrintCallback(freq=1000)]
    callbacks = [callbacks.PrintCallback(freq=100), ]
    # plotter = None
    # agent = agents.OffPolicyContinuousAgent(ddpg, policy, callbacks, args)
    # TODO: debug a2c
    a2c = algorithms.A2C(
        policy_network=policy_network,
        value_network=value_network,
        policy=policy,
        device=args.device,
        gamma=args.gamma,
        entropy_weighting=0.001)
    agent = agents.MultithreadedOnPolicyDiscreteAgent(
        algorithm=a2c,
        policy=policy,
        callbacks=callbacks,
        args=args)
    # preprocess = transforms.Compose([utils.Resize((64, 64)), utils.ImgToTensor()])
    preprocess = lambda x: torch.from_numpy(x).float()
    agent.train(make_env, preprocess)

# TODO:
# 1. Reverify that it works by resetting everything to normala - yeah something doesn't work anymore - something with the process
# 4. convert memory to a regular list
# 5. convert memory to a pytorch array