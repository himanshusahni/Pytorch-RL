import argparse
import gym
import torch
import torchvision.transforms as transforms
import torch.multiprocessing as mp

from pytorch_rl import networks, utils, callbacks, agents, algorithms, policies

# def preprocess(x):
#     return torch.from_numpy(x).float()


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
    parser.add_argument('--frame_stack',
                        help='how many observations form a state',
                        type=int,
                        default=4)
    parser.add_argument('--gamma',
                        help='discount factor',
                        type=float,
                        default=0.99)
    parser.add_argument('--nb_threads',
                        help='number of processes for multithreaded algorithms',
                        type=int,
                        default=8)
    parser.add_argument('--nb_rollout_steps',
                        help='steps per rollout for AC type algorithms',
                        type=int,
                        default=64)
    parser.add_argument('--test_freq',
                        help='how often to test',
                        type=int,
                        default=1)
    parser.add_argument('--env',
                        help='environment name',)
    args = parser.parse_args()

    mp.set_start_method("spawn")

    args.env = 'Breakout-v4'
    # args.env = 'LunarLander-v2'
    def make_env():
        return gym.make(args.env)
    env = make_env()

    observation_space = env.observation_space
    observation_space.shape = (3, args.obs_size, args.obs_size)
    action_space = env.action_space.n
    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # args.device = torch.device("cpu")

    policy_network = networks.ConvPolicy_64x64
    value_network = networks.ConvValue_64x64
    # policy_network = networks.FCPolicy
    # value_network = networks.FCValue
    # ddpg = algorithms.TD3(policy_network, value_network, args)
    # policy = policies.OrnsteinUhlenbeckPolicy(env.action_space.low.shape)
    policy = policies.MultinomialPolicy()
    # plotter = VisdomLinePlotter(env_name='ddpg')
    # callbacks = [callbacks.PlotCallback(plotter, freq=100), callbacks.PrintCallback(freq=1000)]
    callbacks = [callbacks.PrintCallback(freq=1), ]
    # plotter = None
    # agent = agents.OffPolicyContinuousAgent(ddpg, policy, callbacks, args)
    if len(observation_space.shape) == 1:
        state_shape = (observation_space.shape[0] * args.frame_stack,)
    else:
        state_shape = (observation_space.shape[0] * args.frame_stack,
                       *observation_space.shape[1:])
    # ppo = algorithms.PPO(
    #     policy_arch=policy_network,
    #     value_arch=value_network,
    ppo = algorithms.PPO(
        actor_critic_arch=networks.ConvTrunk_64x64,
        state_shape=state_shape,
        action_space=action_space,
        policy=policy,
        ppo_epochs=50,
        clip_param=0.2,
        target_kl=0.01,
        minibatch_size=32,
        device=args.device,
        gamma=args.gamma,
        lam=0.95,
        clip_value_loss=True,
        value_loss_weighting=0.5,
        entropy_weighting=0.01)
    agent = agents.MultithreadedOnPolicyDiscreteAgent(
        algorithm=ppo,
        policy=policy,
        nb_rollout_steps=args.nb_rollout_steps,
        state_shape=state_shape,
        max_trains_steps=args.max_train_steps,
        test_freq=args.test_freq,
        nb_threads=args.nb_threads,
        frame_stack=args.frame_stack,
        device=args.device,
        callbacks=callbacks)
    preprocess = transforms.Compose([utils.Resize((64, 64)), utils.ImgToTensor()])
    agent.train(make_env, preprocess)

# TODO:
# 1. Reverify that it works by resetting everything to normala - yeah something doesn't work anymore - something with the process
# 4. convert memory to a regular list
# 5. convert memory to a pytorch array