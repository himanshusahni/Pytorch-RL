import argparse
import gym
import torch
import torchvision.transforms as transforms
import torch.multiprocessing as mp


from pytorch_rl import networks, utils, callbacks, agents, algorithms, policies

if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--max_train_steps',
                        help='maximum environment steps allowed for training',
                        type=int,
                        default=100000)
    parser.add_argument('--obs_size',
                        help='resize observations from environment',
                        type=int,
                        default=64)
    parser.add_argument('--gamma',
                        help='discount factor',
                        type=float,
                        default=0.99)
    parser.add_argument('--entropy_weighting',
                        help='entropy loss contribution',
                        type=float,
                        default=0.05)
    parser.add_argument('--nb_threads',
                        help='number of processes for collecting experience',
                        type=int,
                        default=5)
    parser.add_argument('--nb_rollout_steps',
                        help='steps per rollout for AC',
                        type=int,
                        default=10)
    parser.add_argument('--test_freq',
                        help='testing frequency',
                        type=int,
                        default=100)
    parser.add_argument('--env',
                        help='environment name',)
    args = parser.parse_args()

    # first enable cuda memory sharing
    mp.set_start_method('spawn')

    # env = ScaledEnv(gym.make('Breakout-v4'))
    args.env = 'Breakout-v4'
    def make_env():
        return gym.make(args.env)
    env = make_env()

    args.observation_space = (3, args.obs_size, args.obs_size)
    args.action_space = env.action_space.n
    # args.device = torch.device("cpu")
    args.device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

    policy_network = networks.ConvPolicy_64x64(args.action_space)
    value_network = networks.ConvValue_64x64()
    policy = policies.MultinomialPolicy
    # plotter = VisdomLinePlotter(env_name='ddpg')
    # callbacks = [callbacks.PlotCallback(plotter, freq=100), callbacks.PrintCallback(freq=1000)]
    callbacks = [callbacks.PrintCallback(freq=100), ]
    a2c = algorithms.A2C(policy_network, value_network, args)
    agent = agents.MultithreadedOnPolicyDiscreteAgent(
        algorithm=a2c,
        policy=policy,
        callbacks=callbacks,
        args=args)
    preprocess = transforms.Compose([utils.Resize((64, 64)), utils.ImgToTensor()])
    agent.train(make_env, preprocess)
    # TODO: verify on an easy environment.
