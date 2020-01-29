import torch.multiprocessing as mp
from pytorch_rl.utils import *

class OffPolicyContinuousAgent:
    def __init__(self, algorithm, policy, callbacks, args):
        self.algorithm = algorithm
        self.policy = policy
        self.args = args
        self.callbacks = callbacks

    def train(self, env):
        #TODO: preprocessing is also now passed here
        #TODO: network now only returns logits. the job of nonlinearity needs to lie with the policy
        #TODO: the environment is passed as a make_env method now
        # first fill up replay buffer to certain length with random actions
        done = True
        ep = -1
        step = 0
        while step < self.args.replay_warmup:
            if done:
                ep += 1
                ep_reward = 0.
                state = torch.from_numpy(env.reset()).float()
            a = env.action_space.sample()
            next_state, r, done, info = env.step(a)
            step += 1
            ep_reward += r
            # update
            a, next_state = torch.from_numpy(a).float(), torch.from_numpy(next_state).float()
            r, done = torch.tensor([r,]).float(), torch.tensor([float(done),]).float()
            self.algorithm.update((state, a, next_state, r, done), train=False)
            state = next_state

        # now start training
        done = True
        metrics = {'ep_reward': AverageMeter(reset_freq=10000),
                   'q_val': AverageMeter(),
                   'bellman_error': AverageMeter(),
                   'policy_loss': AverageMeter()}

        while step < self.args.max_train_steps:
            if done:
                if ep >= 0:
                    metrics['ep_reward'].update(ep_reward)
                ep += 1
                ep_reward = 0.
                state = torch.from_numpy(env.reset()).float()
                # callbacks
                metrics['ep'] = ep
                for callback in self.callbacks:
                    callback.on_episode_end(metrics)

            mean = self.algorithm.pi(torch.unsqueeze(state, 0).to(self.args.device))
            a = np.squeeze(self.policy(mean.detach().cpu().numpy()))
            next_state, r, done, info = env.step(a)
            step += 1
            ep_reward += r
            # update
            next_state, a = torch.from_numpy(next_state).float(), torch.from_numpy(a).float()
            r, done = torch.tensor([r,]).float(), torch.tensor([float(done),]).float()
            self.algorithm.update((state, a, next_state, r, done), metrics=metrics)
            state = next_state
            # callbacks
            metrics['step'] = step
            for callback in self.callbacks:
                callback.on_step_end(metrics)


class MultithreadedOnPolicyDiscreteAgent:
    def __init__(self, algorithm, policy, callbacks, args):
        self.algorithm = algorithm
        self.policy = policy
        self.args = args
        self.callbacks = callbacks

    class Clone:
        def __init__(self, t, make_env, preprocess, policy, nb_rollout, device, rollout):
            """create a new environment"""
            self.t = t
            self.env = make_env()
            self.nb_rollout = nb_rollout
            self.rollout = rollout
            self.preprocess = preprocess
            self.policy = policy
            self.device = device

        def run(self, pi, startq, stopq):
            done = True
            while True:
                startq.get()
                for step in range(self.nb_rollout):
                    if done:
                        state = self.preprocess(self.env.reset())
                    state = state.to(self.device)
                    action = self.policy(pi(state.unsqueeze(dim=0)))
                    next_state, r, done, _ = self.env.step(action.detach().cpu().numpy())
                    self.rollout['states'][self.t, step] = state
                    self.rollout['actions'][self.t, step] = action
                    self.rollout['rewards'][self.t, step] = r
                    self.rollout['dones'][self.t, step] = float(done)
                    state = self.preprocess(next_state)
                # finally add the next state into the states buffer as well to do value estimation
                self.rollout['states'][self.t, self.nb_rollout] = state.to(self.device)
                stopq.put(self.t)

        def test_run(self, pi, rq):
            while True:
                idx = rq.get()
                state = self.preprocess(self.env.reset())
                ep_reward = 0
                done = False
                while not done:
                    state = state.to(self.device)
                    action = self.policy(pi(state.unsqueeze(dim=0)))
                    next_state, r, done, _ = self.env.step(action.detach().cpu().numpy())
                    ep_reward += r
                    state = self.preprocess(next_state)
                print("Testing reward {:.3f}".format(ep_reward))
                self.rollout['test_reward'][idx] = ep_reward

    def train(self, make_env, preprocess):
        # create shared data between agent clones
        rollout = {
            'states': torch.empty(
                self.args.nb_threads,
                self.args.nb_rollout_steps+1,
                *self.args.observation_space).to(self.args.device).share_memory_(),
            'actions': torch.empty(
                self.args.nb_threads,
                self.args.nb_rollout_steps,
                dtype=torch.long).to(self.args.device).share_memory_(),
            'rewards': torch.empty(
                self.args.nb_threads,
                self.args.nb_rollout_steps).to(self.args.device).share_memory_(),
            'dones': torch.empty(
                self.args.nb_threads,
                self.args.nb_rollout_steps).to(self.args.device).share_memory_(),
            'test_reward': torch.empty(self.args.max_train_steps//self.args.test_freq).share_memory_()
        }
        # stopqs and startqs tell when the actors should collect rollouts
        stopqs = []
        startqs = []
        # make the policy available to all processes
        self.algorithm.pi.share_memory()
        procs = []
        # clones = []
        for t in range(self.args.nb_threads):
            startq = mp.Queue(1)
            startqs.append(startq)
            stopq = mp.Queue(1)
            stopqs.append(stopq)
            c = self.Clone(
                t, make_env, preprocess, self.policy, self.args.nb_rollout_steps, self.args.device, rollout)
            # clones.append(c)
            proc = mp.Process(target=c.run, args=(self.algorithm.pi, startq, stopq))
            procs.append(proc)
            proc.start()
        # have a thread for testing
        testq = mp.Queue(1)
        test_clone = self.Clone(
                t+1, make_env(), preprocess, self.policy, self.args.nb_rollout_steps, self.args.device, rollout)
        test_proc = mp.Process(target=test_clone.test_run, args=(self.algorithm.pi, testq))
        test_proc.start()

        # train
        step = 0
        metrics = {'policy_loss': AverageMeter(),
                   'val_loss': AverageMeter(),
                   'policy_entropy': AverageMeter(),
                   'avg_val': AverageMeter(),
                   'avg_reward': AverageMeter()}
        while step < self.args.max_train_steps:
            # start collecting data
            for start in startqs:
                start.put(1)
            # wait for the agents to finish getting data
            for stop in stopqs:
                stop.get()
            # for c in clones:
            #     c.run(self.al
            # update!
            self.algorithm.update((
                rollout['states'][:,:-1],
                rollout['actions'],
                rollout['rewards'],
                rollout['dones']), rollout['states'][:,-1], metrics)
            step += 1
            # callbacks
            metrics['step'] = step
            # TODO: do we have to clone this??
            # metrics['states'] = rollout['states'].clone().cpu()
            # metrics['actions'] = rollout['actions'].clone().cpu()
            # metrics['rewards'] = rollout['rewards'].clone().cpu()
            # metrics['dones'] = rollout['dones'].clone().cpu()
            for callback in self.callbacks:
                callback.on_step_end(metrics)
            # test
            if step % self.args.test_freq == 0:
                testq.put(step // self.args.test_freq)
                # test_clone.test_run(self.algorithm.pi, testq)

        # end by shutting down processes
        for p in procs:
            p.terminate()
        test_proc.terminate()

