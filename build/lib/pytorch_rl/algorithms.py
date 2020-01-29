from pytorch_rl.networks import *
from pytorch_rl.utils import *

class DDPG():
    #TODO: just make it take in policy and value networks and create deep copies of them for target
    def __init__(self, policy_arch, value_arch, args):
        # create Q value and policy and target networks
        self.Q = value_arch(args.observation_space.shape[0], args.action_space.shape[0])
        self.target_Q = value_arch(args.observation_space.shape[0], args.action_space.shape[0])
        self.Q.to(args.device)
        self.target_Q.to(args.device)
        for target_param, param in zip(self.target_Q.parameters(), self.Q.parameters()):
            target_param.data.copy_(param.data)
        self.Q_optimizer = torch.optim.Adam(self.Q.parameters(), lr=1e-3)
        self.pi = policy_arch(args.observation_space.shape[0], args.action_space.shape[0])
        self.target_pi = policy_arch(args.observation_space.shape[0], args.action_space.shape[0])
        self.pi.to(args.device)
        self.target_pi.to(args.device)
        for target_param, param in zip(self.target_pi.parameters(), self.pi.parameters()):
            target_param.data.copy_(param.data)
        self.pi_optimizer = torch.optim.Adam(self.pi.parameters(), lr=1e-4)

        self.replay_buffer = ExperienceReplay(args.replay_capacity)
        self.mse = nn.MSELoss()
        self.args = args
        self.polyak_constant = 0.001

    def copy_target(self):
        """
        move target networks closer to training networks
        """
        for target_param, param in zip(self.target_pi.parameters(), self.pi.parameters()):
            target_param.data.copy_(
                self.polyak_constant * param.data + (1.0 - self.polyak_constant) * target_param.data)
        for target_param, param in zip(self.target_Q.parameters(), self.Q.parameters()):
            target_param.data.copy_(
                self.polyak_constant * param.data + (1.0 - self.polyak_constant) * target_param.data)

    def update(self, exp, train=True, metrics=None):
        self.replay_buffer.append(exp)
        if train:
            # sample a batch from the replay
            state_batch, action_batch, next_state_batch, reward_batch, done_batch =\
                self.replay_buffer.sample(self.args.batch_size)
            state_batch = state_batch.to(self.args.device)
            action_batch = action_batch.to(self.args.device)
            next_state_batch = next_state_batch.to(self.args.device)
            reward_batch = reward_batch.to(self.args.device)
            done_batch = done_batch.to(self.args.device)

            # first train the critic
            self.Q_optimizer.zero_grad()
            next_Q = self.target_Q(next_state_batch, self.target_pi(next_state_batch))
            next_Q = torch.squeeze(next_Q).detach()
            y = reward_batch + self.args.gamma * (1 - done_batch) * next_Q
            this_Q = torch.squeeze(self.Q(state_batch, action_batch))
            bellman_error = self.mse(this_Q, y)
            bellman_error.backward()
            self.Q_optimizer.step()
            # now train the actor
            self.pi_optimizer.zero_grad()
            policy_loss = - self.Q(state_batch, self.pi(state_batch)).mean()
            policy_loss.backward()
            self.pi_optimizer.step()
            # update the target networks
            self.copy_target()
            metrics['bellman_error'].update(bellman_error.item())
            metrics['policy_loss'].update(policy_loss.item())
            metrics['q_val'].update(this_Q.mean().item())


class TD3(DDPG):
    """twin-delayed ddpg"""
    def __init__(self, policy_arch, value_arch, args):
        super().__init__(policy_arch, value_arch, args)
        # additional stuff for TD3 part
        self.policy_delay = 2
        self.updates = 0
        self.Q2 = value_arch(args.observation_space.shape[0], args.action_space.shape[0])
        self.target_Q2 = value_arch(args.observation_space.shape[0], args.action_space.shape[0])
        self.Q2.to(args.device)
        self.target_Q2.to(args.device)
        for target_param, param in zip(self.target_Q2.parameters(), self.Q2.parameters()):
            target_param.data.copy_(param.data)
        self.Q2_optimizer = torch.optim.Adam(self.Q2.parameters(), lr=1e-3)

    def copy_target(self):
        super().copy_target()
        # also copy over the second Q network
        for target_param, param in zip(self.target_Q2.parameters(), self.Q2.parameters()):
            target_param.data.copy_(
                self.polyak_constant * param.data + (1.0 - self.polyak_constant) * target_param.data)

    def update(self, exp, train=True, metrics=None):
        self.replay_buffer.append(exp)
        if train:
            # sample a batch from the replay
            state_batch, action_batch, next_state_batch, reward_batch, done_batch = \
                self.replay_buffer.sample(self.args.batch_size)
            state_batch = state_batch.to(self.args.device)
            action_batch = action_batch.to(self.args.device)
            next_state_batch = next_state_batch.to(self.args.device)
            reward_batch = reward_batch.to(self.args.device)
            done_batch = done_batch.to(self.args.device)

            # first train the critic
            self.Q_optimizer.zero_grad()
            # calculate a smoothed optimal action
            noise = torch.from_numpy(0.1 * np.random.randn(*tuple(action_batch.size())))
            noise = noise.clamp(-0.2, 0.2).float().to(self.args.device)
            smoothed_action = (self.target_pi(next_state_batch) + noise).clamp(-1., 1.)
            # of the two Q networks, find which one gives lower Q value
            next_Q1 = self.target_Q(next_state_batch, smoothed_action)
            next_Q2 = self.target_Q2(next_state_batch, smoothed_action)
            next_Q = torch.min(next_Q1, next_Q2)
            next_Q = torch.squeeze(next_Q).detach()
            # create target and error for both Q networks
            y = reward_batch + self.args.gamma * (1 - done_batch) * next_Q
            this_Q1 = torch.squeeze(self.Q(state_batch, action_batch))
            this_Q2 = torch.squeeze(self.Q2(state_batch, action_batch))
            bellman_error = self.mse(this_Q1, y) + self.mse(this_Q2, y)
            bellman_error.backward()
            # update both Q networks
            self.Q_optimizer.step()
            self.Q2_optimizer.step()
            # train the actor and target networks at a delay
            if self.updates % self.policy_delay == 0:
                self.pi_optimizer.zero_grad()
                policy_loss = - self.Q(state_batch, self.pi(state_batch)).mean()
                policy_loss.backward()
                self.pi_optimizer.step()
                # update the target networks
                self.copy_target()
                metrics['policy_loss'].update(policy_loss.item())
            metrics['bellman_error'].update(bellman_error.item())
            metrics['q_val'].update(next_Q.mean().item())
            self.updates += 1


#TODO: entropy loss now calculated directly within policy for logits
class A2C():
    """Advantage actor critic algorithm"""
    def __init__(self, policy_network, value_network, policy, device, gamma, entropy_weighting):
        # create value and policy networks
        self.V = value_network
        self.V.to(device)
        self.V_optimizer = torch.optim.Adam(self.V.parameters(), lr=1e-3)
        self.pi = policy_network
        self.pi.to(device)
        self.pi_optimizer = torch.optim.Adam(self.pi.parameters(), lr=1e-4)
        self.policy = policy
        self.mse = nn.MSELoss()
        self.device = device
        self.gamma = gamma
        self.entropy_weighting = entropy_weighting

    def save(self, path):
        torch.save({'V':self.V, 'pi':self.pi}, path)

    def update(self, exp, final_states, metrics=None):
        # states = torch.cat([e[0].unsqueeze(1) for e in exp[:-1]], dim=1)
        # logits = torch.cat([e[1].unsqueeze(1) for e in exp[:-1]], dim=1)
        # actions = torch.cat([e[2].unsqueeze(1) for e in exp[:-1]], dim =1)
        # rewards = torch.cat([e[3].unsqueeze(1) for e in exp[:-1]], dim =1)
        # dones = torch.cat([e[4].unsqueeze(1) for e in exp[:-1]], dim =1)
        states = exp[0]
        actions = exp[1]
        rewards = exp[2]
        dones = exp[3]
        if metrics is not None:
            metrics['avg_reward'].update(rewards.mean().item())

        # estimate values of final states in rollout
        final_values = self.V(final_states).detach().squeeze()
        # if rollout ended in terminal state, then the value is zero
        final_values *= (1 - dones[:,-1])
        # calculate returns from rewards in place
        rewards = torch.cat((rewards, final_values.unsqueeze(dim=1)), dim=1)
        # now start summing backwards, zeroing out returns for terminal states
        for i in reversed(range(1, rewards.size(1))):
            rewards[:, i-1] += self.gamma * rewards[:, i] * (1-dones[:, i-1])
        # discard the last bootstrapping value
        rewards = rewards[:, :-1].flatten()

        batchsize = states.size(0)
        steps = states.size(1)
        states = states.reshape(batchsize * steps, *states.size()[2:])
        values = self.V(states).squeeze()
        # calculate advantage
        adv = rewards - values

        # redo some forward pass on policy to get loss
        logits = self.pi(states)
        logprobs = self.policy.logprobs(logits, actions.view(batchsize * steps, -1))

        # policy and value losses
        policy_loss = -(logprobs * adv.detach()).mean()
        entropy_loss = -self.policy.entropy(logits).mean()
        value_loss = adv.pow(2).mean()

        self.V_optimizer.zero_grad()
        self.pi_optimizer.zero_grad()
        total_loss = policy_loss + value_loss + self.entropy_weighting * entropy_loss
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.V.parameters(), 1)
        torch.nn.utils.clip_grad_norm_(self.pi.parameters(), 1)
        self.V_optimizer.step()
        self.pi_optimizer.step()

        if metrics is not None:
            # bookkeeping
            metrics['policy_loss'].update(policy_loss.item())
            metrics['val_loss'].update(value_loss.item())
            metrics['policy_entropy'].update(-entropy_loss.item())
            metrics['avg_val'].update(values.mean().item())
        return total_loss.item()
