from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler

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

    def update(self, exp, metrics=None):
        state_batch, action_batch, next_state_batch, reward_batch, done_batch = exp
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
        # sample a batch from the replay
        state_batch, action_batch, next_state_batch, reward_batch, done_batch = exp

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

class DQN():
    """Deep Q Network from Mnih et al. (Nature 2015)"""
    def __init__(self, q_arch, channels, policy, gamma, device):
        self.q = q_arch(channels=channels).to(device)
        self.target_q = q_arch(channels=channels).to(device)
        self.copy_target()
        self.policy = policy
        self.gamma = gamma
        self.q_optimizer = torch.optim.Adam(self.q.parameters(), lr=1e-4)
        self.polyak_constant = 0.001

    def copy_target(self):
        """
        copy over weights from training network into target network
        """
        for target_param, param in zip(self.target_q.parameters(), self.q.parameters()):
            target_param.data.copy_(param.data)

    def move_target(self):
        """
        move target network closer to training network
        """
        for target_param, param in zip(self.target_q.parameters(), self.q.parameters()):
            target_param.data.copy_(
                self.polyak_constant * param.data + (1.0 - self.polyak_constant) * target_param.data)

    def update(self, exp, metrics=None, scope=''):
        states = exp[0]
        actions = exp[1]
        next_states = exp[2]
        rewards = exp[3]
        dones = exp[4]
        if metrics is not None:
            metrics[scope+'/avg_reward'].update(rewards.mean().item())
        batchsize = states.size(0)

        # calculate target DDQN style
        target_qvals = self.target_q(next_states).detach()
        next_qvals = self.q(next_states).detach()
        argmax_a = next_qvals.max(dim=1)[1]
        target_qvals = target_qvals[range(batchsize), argmax_a]
        target = rewards + (1 - dones) * self.gamma * target_qvals

        # loss and step!
        qvals = self.q(states)
        entropy = self.policy.entropy(qvals).mean()
        y = qvals[range(batchsize), actions]
        loss = F.smooth_l1_loss(y, target).mean()  # huber loss
        self.q_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q.parameters(), 0.5)
        self.q_optimizer.step()

        # target update!
        self.move_target()

        if metrics is not None:
            # bookkeeping
            metrics[scope+'/loss'].update(loss.item())
            metrics[scope+'/avg_val'].update(qvals.max(dim=1)[0].mean().item())
            metrics[scope+'/policy_entropy'].update(entropy.item())
        return loss


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

    def update(self, exp, final_states, step=True, metrics=None, old_logits=None, scope=''):
        states = exp[0]
        actions = exp[1]
        rewards = exp[2]
        dones = exp[3]
        if metrics is not None:
            metrics[scope+'/avg_reward'].update(rewards.mean().item())

        # estimate values of final states in rollout
        final_values = self.V(final_states).detach().squeeze(dim=1)
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

        # importance sampling
        if old_logits is not None:
            new_logprobs = logprobs.view(batchsize, steps).detach()
            cum_logprobs = torch.zeros(batchsize, steps).to(self.device)
            cum_logprobs[:, steps-1] = new_logprobs[:, steps-1]
            for j in reversed(range(steps-1)):
                cum_logprobs[:, j] = (1 - dones[:, j]) * cum_logprobs[:, j+1] + new_logprobs[:, j]
            old_logits = old_logits.reshape(batchsize * steps, *old_logits.size()[2:])
            old_logprobs = self.policy.logprobs(old_logits, actions.view(batchsize * steps, -1))
            old_logprobs = old_logprobs.view(batchsize, steps).detach()
            cum_oldlogprobs = torch.zeros(batchsize, steps).to(self.device)
            cum_oldlogprobs[:, steps-1] = old_logprobs[:, steps-1]
            for j in reversed(range(steps-1)):
                cum_oldlogprobs[:, j] = (1 - dones[:, j]) * cum_oldlogprobs[:, j+1] + old_logprobs[:, j]
            # clamp to avoid problems
            ratios = torch.clamp(cum_logprobs - cum_oldlogprobs, min=-3, max=3).exp()
            ratios = ratios.view(batchsize * steps)
            adv = adv * ratios
            metrics[scope+'/IS_ratio'].update(ratios.mean().item())

        # policy and value losses
        policy_loss = -(logprobs * adv.detach()).mean()
        entropy_loss = -self.policy.entropy(logits).mean()
        value_loss = adv.pow(2).mean()

        total_loss = policy_loss + value_loss + self.entropy_weighting * entropy_loss
        if step:
            self.V_optimizer.zero_grad()
            self.pi_optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.V.parameters(), 0.5)
            torch.nn.utils.clip_grad_norm_(self.pi.parameters(), 0.5)
            self.V_optimizer.step()
            self.pi_optimizer.step()

        if metrics is not None:
            # bookkeeping
            metrics[scope+'/policy_loss'].update(policy_loss.item())
            metrics[scope+'/val_loss'].update(value_loss.item())
            metrics[scope+'/policy_entropy'].update(-entropy_loss.item())
            metrics[scope+'/avg_val'].update(values.mean().item())
        return total_loss


class PPO():
    """Advantage actor critic algorithm"""
    def __init__(self, actor_critic_arch, trunk_arch, state_shape, action_space,
                 policy, ppo_epochs, clip_param, target_kl,
                 minibatch_size, clip_value_loss, device, gamma, lam,
                 value_loss_weighting, entropy_weighting):
        self.actor_critic = actor_critic_arch(trunk_arch, state_shape, action_space)
        self.new_actor_critic = actor_critic_arch(trunk_arch, state_shape, action_space)
        self.actor_critic.to(device)
        self.new_actor_critic.to(device)
        self.copy_target()
        # self.V_optimizer = torch.optim.Adam(self.new_actor_critic.V_parameters(), lr=1e-4)
        # self.pi_optimizer = torch.optim.Adam(self.new_actor_critic.pi_parameters(), lr=1e-4)
        self.lr = 2.5e-4
        self.optimizer = torch.optim.Adam(self.new_actor_critic.parameters(), lr=self.lr, eps=1e-5)
        self.pi = self.actor_critic.pi
        self.V = self.actor_critic.V
        self.policy = policy
        self.ppo_epochs = ppo_epochs
        self.minibatch_size = minibatch_size
        self.clip_param = clip_param
        self.clip_param_now = clip_param
        self.target_kl = target_kl
        self.clip_value_loss = clip_value_loss
        self.mse = nn.MSELoss()
        self.device = device
        self.gamma = gamma
        self.lam = lam
        self.value_loss_weighting = value_loss_weighting
        self.entropy_weighting = entropy_weighting

    def save(self, path):
        torch.save({'AC': self.actor_critic}, path)

    def copy_target(self):
        """
        copy over policy and value weights to old network
        """
        for target_param, param in zip(self.actor_critic.parameters(), self.new_actor_critic.parameters()):
            target_param.data.copy_(param.data)

    def anneal(self, step, max_steps):
        """perform annealing as needed"""
        self.clip_param_now -= self.clip_param / max_steps
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.lr * (1 - step/max_steps)

    def update(self, exp, final_states, metrics=None, scope=''):
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
            metrics[scope+'/avg_reward'].update(rewards.mean().item())

        num_procs = states.size(0)
        steps = states.size(1)
        states = states.reshape(num_procs * steps, *states.size()[2:])
        actions = actions.view(num_procs * steps, -1)

        logits, values = self.actor_critic(states)
        logits = logits.detach()
        values = values.detach()

        # old policy logprobs
        logprobs = self.policy.logprobs(logits, actions)

        # estimate values of final states in rollout
        final_values = self.V(final_states).detach().squeeze(dim=1)
        # if rollout ended in terminal state, then the value is zero
        final_values *= (1 - dones[:,-1])
        # calculate returns and generalized advantage from rewards
        values_reshaped = values.view(num_procs, steps)
        values_reshaped = torch.cat((values_reshaped, final_values.unsqueeze(dim=1)), dim=1)
        adv = torch.zeros_like(values_reshaped)
        # now start summing backwards, zeroing out returns for terminal states
        for i in reversed(range(rewards.size(1))):
            delta = rewards[:, i] + self.gamma * values_reshaped[:, i+1] * (1 - dones[:, i]) - values_reshaped[:, i]
            adv[:, i] = delta + self.gamma * self.lam * adv[:, i+1] * (1 - dones[:, i])

        # remove final bootstrapping values
        adv = adv[:, :-1]
        values_reshaped = values_reshaped[:, :-1]
        # get returns
        returns = (values_reshaped + adv).flatten()

        # standardize advantage
        adv = (adv - adv.mean(dim=1, keepdim=True)) / (adv.std(dim=1, keepdim=True) + 1e-5)
        adv = adv.flatten()

        total_policy_loss = 0
        total_entropy_loss = 0
        total_value_loss = 0
        nb_updates = 0

        for epoch in range(self.ppo_epochs):
            # sample minibatch indices
            total_samples = states.size(0)
            sampler = BatchSampler(
                SubsetRandomSampler(range(total_samples)),
                self.minibatch_size,
                drop_last=True)
            total_kl = 0
            nb_minibatches = 0
            for indices in sampler:
                states_batch = states[indices]
                actions_batch = actions[indices]
                adv_batch = adv[indices]
                oldvalues_batch = values[indices]
                old_logprobs_batch = logprobs[indices]
                old_logits_batch = logits[indices]
                returns_batch = returns[indices]
                # redo some forward pass on new policy
                newlogits, new_values = self.new_actor_critic(states_batch)
                # clipped policy loss
                newlogprobs = self.policy.logprobs(newlogits, actions_batch)
                ratio = torch.exp(newlogprobs - old_logprobs_batch)
                loss1 = ratio * adv_batch
                loss2 = torch.clamp(
                    ratio, 1. - self.clip_param_now, 1. + self.clip_param_now) * adv_batch
                policy_loss = -torch.min(loss1, loss2).mean()
                # entropy
                entropy = self.policy.entropy(newlogits)
                entropy_loss = -entropy.mean()
                # clipped value losses
                value_losses = (new_values - returns_batch).pow(2)
                if not self.clip_value_loss:
                    value_loss = 0.5 * value_losses.mean()
                else:
                    values_clipped = oldvalues_batch + torch.clamp(
                        new_values - oldvalues_batch, -self.clip_param_now, self.clip_param_now)
                    clipped_value_losses = (values_clipped - returns_batch).pow(2)
                    value_loss = 0.5 * torch.max(value_losses, clipped_value_losses).mean()
                # value_loss = value_losses.mean()
                # update!
                # self.V_optimizer.zero_grad()
                # self.pi_optimizer.zero_grad()
                self.optimizer.zero_grad()
                total_loss = policy_loss + \
                             self.value_loss_weighting * value_loss + \
                             self.entropy_weighting * entropy_loss
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.new_actor_critic.parameters(), 0.5)
                # self.V_optimizer.step()
                # self.pi_optimizer.step()
                self.optimizer.step()
                # check kl for early stopping
                kl = - (self.policy.probs(newlogits) * self.policy.logprobs(old_logits_batch)).sum(dim=1) - entropy
                # bookkeeping
                total_kl += kl.mean().item()
                total_policy_loss += policy_loss.item()
                total_entropy_loss += entropy_loss.item()
                total_value_loss += self.value_loss_weighting * value_loss.item()
                nb_updates += 1
                nb_minibatches += 1
            # early stop if mean kl is above threshold
            if total_kl / nb_minibatches > 1.5 * self.target_kl:
                break

        total_policy_loss /= nb_updates
        total_entropy_loss /= nb_updates
        total_value_loss /= nb_updates
        # copy over new parameters into policy and value networks
        self.copy_target()
        if metrics is not None:
            # bookkeeping
            metrics[scope+'/policy_loss'].update(total_policy_loss)
            metrics[scope+'/val_loss'].update(total_value_loss)
            metrics[scope+'/policy_entropy'].update(-total_entropy_loss)
            metrics[scope+'/avg_val'].update(values.mean().item())
        return (total_policy_loss + self.value_loss_weighting * total_value_loss +
                self.entropy_weighting * total_entropy_loss)
