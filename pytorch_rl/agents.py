import torch.multiprocessing as mp
from pytorch_rl.utils import *


class OffPolicyAgent:
    def __init__(self, algorithm, policy, callbacks, args):
        self.algorithm = algorithm
        self.policy = policy
        self.args = args
        self.callbacks = callbacks
        self.replay_buffer = ExperienceReplay(args.replay_capacity)

    def train(self, make_env, preprocess):
        #TODO: network now only returns logits. the job of nonlinearity needs to lie with the policy
        # first fill up replay buffer to certain length with random actions
        env = make_env()
        done = True
        ep = -1
        step = 0
        while step < self.args.replay_warmup:
            if done:
                ep += 1
                ep_reward = 0.
                state = preprocess(env.reset())
            a = env.action_space.sample()
            next_state, r, done, info = env.step(a)
            next_state = preprocess(next_state)
            step += 1
            ep_reward += r
            # update
            a, r, done = torch.from_numpy(a).float(), torch.tensor([r,]).float(), torch.tensor([float(done),]).float()
            self.replay_buffer.append((state, a, next_state, r, done))
            state = next_state

        # now start training
        done = True
        metrics = {'ep_reward': AverageMeter(),
                   'q_val': AverageMeter(),
                   'bellman_error': AverageMeter(),
                   'policy_loss': AverageMeter()}

        while step < self.args.max_train_steps:
            if done:
                if ep >= 0:
                    metrics['ep_reward'].update(ep_reward)
                ep += 1
                ep_reward = 0.
                state = preprocess(env.reset())
                # callbacks
                metrics['ep'] = ep
                for callback in self.callbacks:
                    callback.on_episode_end(metrics)

            logits = self.algorithm.pi(torch.unsqueeze(state, 0).to(self.args.device))
            a = np.squeeze(self.policy(logits.detach().cpu().numpy()))
            next_state, r, done, info = env.step(a)
            next_state = preprocess(next_state)
            step += 1
            ep_reward += r
            # update
            a, r, done = torch.from_numpy(a).float(), torch.tensor([r,]).float(), torch.tensor([float(done),]).float()
            self.replay_buffer.append((state, a, next_state, r, done))
            state = next_state
            # sample a batch from the replay
            state_batch, action_batch, next_state_batch, reward_batch, done_batch = \
                self.replay_buffer.sample(self.args.batch_size)
            state_batch = state_batch.to(self.args.device)
            action_batch = action_batch.to(self.args.device)
            next_state_batch = next_state_batch.to(self.args.device)
            reward_batch = reward_batch.to(self.args.device)
            done_batch = done_batch.to(self.args.device)
            self.algorithm.update((state_batch, action_batch, next_state_batch, reward_batch, done_batch))
            # callbacks
            metrics['step'] = step
            for callback in self.callbacks:
                callback.on_step_end(metrics)


class MultithreadedRolloutAgent:
    """
    Used mainly to collect rollout data in an environment.
    If policy is not given, acts randomly
    """
    def __init__(self, nb_threads, nb_rollout_steps, max_env_steps, state_shape,
                 frame_stack, device, callbacks, pi=None, policy=None):
        self.nb_threads = nb_threads
        self.nb_rollout_steps = nb_rollout_steps
        self.max_env_steps = max_env_steps
        self.max_train_steps = int(max_env_steps // (nb_threads * nb_rollout_steps))
        self.state_shape = state_shape
        self.k = frame_stack
        self.device = device
        self.callbacks = callbacks
        self.pi = pi
        self.policy = policy

    class Clone:
        def __init__(self, t, preprocess, nb_rollout, k, device, rollout, policy):
            """create a new environment"""
            self.t = t
            self.nb_rollout = nb_rollout
            self.rollout = rollout
            self.preprocess = preprocess
            self.k = k
            self.device = device
            self.policy = policy

        def run(self, pi, env, startq, stopq):
            from phys_env import phys_env
            self.env = phys_env.PhysEnv()
            # self.env = env
            # self.env.env = phys_env.PhysEnv()
            done = True
            while True:
                startq.get()
                for step in range(self.nb_rollout):
                    if done:
                        obs = self.preprocess(self.env.reset()).to(self.device)
                        stateq = deque(
                            [torch.zeros(obs.shape).to(self.device)
                             for _ in range(self.k-1)], maxlen=self.k)
                        stateq.append(obs)
                        state = torch.cat(list(stateq), dim=0)
                    if self.policy:
                        action = self.policy(pi(state.unsqueeze(dim=0)))
                        next_obs, r, done, _ = self.env.step(action.detach().cpu().numpy())
                    else:
                        action = self.env.action_space.sample()
                        next_obs, r, done, _ = self.env.step(action)
                    self.rollout['states'][self.t, step] = obs
                    self.rollout['actions'][self.t, step] = action
                    self.rollout['rewards'][self.t, step] = r
                    self.rollout['dones'][self.t, step] = float(done)
                    obs = self.preprocess(next_obs).to(self.device)
                    stateq.append(obs)
                    state = torch.cat(list(stateq), dim=0)
                # finally add the next state into the states buffer as well to do value estimation
                self.rollout['states'][self.t, self.nb_rollout] = obs
                stopq.put(self.t)

    def train(self, make_env, preprocess):
        # create shared data between agent clones
        rollout = {
            'states': torch.empty(
                self.nb_threads,
                self.nb_rollout_steps+1,
                *self.state_shape).to(self.device).share_memory_(),
            'actions': torch.empty(
                self.nb_threads,
                self.nb_rollout_steps,
                dtype=torch.long).to(self.device).share_memory_(),
            'rewards': torch.empty(
                self.nb_threads,
                self.nb_rollout_steps).to(self.device).share_memory_(),
            'dones': torch.empty(
                self.nb_threads,
                self.nb_rollout_steps).to(self.device).share_memory_(),
        }
        # stopqs and startqs tell when the actors should collect rollouts
        stopqs = []
        startqs = []
        procs = []
        for t in range(self.nb_threads):
            startq = mp.Queue(1)
            startqs.append(startq)
            stopq = mp.Queue(1)
            stopqs.append(stopq)
            c = self.Clone(
                t, preprocess, self.nb_rollout_steps, self.k, self.device, rollout, self.policy)
            proc = mp.Process(target=c.run, args=(self.pi, make_env(), startq, stopq))
            procs.append(proc)
            proc.start()

        # train
        step = 0
        while step < self.max_train_steps:
            # start collecting data
            for start in startqs:
                start.put(1)
            # wait for the agents to finish getting data
            for stop in stopqs:
                stop.get()
            # update!
            metrics = {}
            step += 1
            # callbacks
            metrics['states'] = rollout['states'].cpu()
            metrics['actions'] = rollout['actions'].cpu()
            metrics['rewards'] = rollout['rewards'].cpu()
            metrics['dones'] = rollout['dones'].cpu()
            for callback in self.callbacks:
                callback.on_step_end(step, metrics)

        # end by shutting down processes
        for p in procs:
            p.terminate()


class MultithreadedOnPolicyDiscreteAgent:
    def __init__(self, algorithm, policy, nb_threads, nb_rollout_steps,
                 max_env_steps, state_shape, test_freq, frame_stack, device,
                 callbacks):
        self.algorithm = algorithm
        self.policy = policy
        self.nb_threads = nb_threads
        self.nb_rollout_steps = nb_rollout_steps
        self.max_env_steps = max_env_steps
        self.max_train_steps = int(max_env_steps // (nb_threads * nb_rollout_steps))
        self.state_shape = state_shape
        self.test_freq = test_freq
        self.k = frame_stack
        self.device = device
        self.callbacks = callbacks

    class Clone:
        def __init__(self, t, preprocess, policy, nb_rollout, k, device, rollout):
            """create a new environment"""
            self.t = t
            self.nb_rollout = nb_rollout
            self.rollout = rollout
            self.preprocess = preprocess
            self.policy = policy
            self.k = k
            self.device = device

        # def preprocess(self, x):
        #     return x

        def run(self, pi, env, startq, stopq):
            from phys_env import phys_env
            # self.env = phys_env.PhysEnv()
            self.env = env
            self.env.env = phys_env.PhysEnv()
            done = True
            while True:
                startq.get()
                for step in range(self.nb_rollout):
                    if done:
                        obs = self.preprocess(self.env.reset()).to(self.device)
                        stateq = deque(
                            [torch.zeros(obs.shape).to(self.device)
                             for _ in range(self.k-1)], maxlen=self.k)
                        stateq.append(obs)
                        state = torch.cat(list(stateq), dim=0)
                    action = self.policy(pi(state.unsqueeze(dim=0))).detach()
                    next_obs, r, done, _ = self.env.step(action.cpu().numpy())
                    self.rollout['states'][self.t, step] = state
                    self.rollout['actions'][self.t, step] = action
                    self.rollout['rewards'][self.t, step] = r
                    self.rollout['dones'][self.t, step] = float(done)
                    obs = self.preprocess(next_obs).to(self.device)
                    stateq.append(obs)
                    state = torch.cat(list(stateq), dim=0)
                # finally add the next state into the states buffer as well to do value estimation
                self.rollout['states'][self.t, self.nb_rollout] = state
                stopq.put(self.t)

        def test_run(self, pi, env, startq, stopq):
            from phys_env import phys_env
            #self.env = phys_env.PhysEnv()
            self.env = env
            self.env.env = phys_env.PhysEnv()
            while True:
                idx = startq.get()
                obs = self.preprocess(self.env.reset())
                stateq = deque(
                    [torch.zeros(obs.shape).to(self.device)
                     for _ in range(self.k - 1)], maxlen=self.k)
                ep_reward = 0
                done = False
                while not done:
                    obs = obs.to(self.device)
                    stateq.append(obs)
                    state = torch.cat(list(stateq), dim=0)
                    action = self.policy(pi(state.unsqueeze(dim=0)), test=True)
                    next_state, r, done, _ = self.env.step(action.detach().cpu().numpy())
                    ep_reward += r
                    obs = self.preprocess(next_state)
                print("Testing reward {:.3f}".format(ep_reward))
                self.rollout['test_reward'][idx] = ep_reward
                stopq.put(1)

    def tosave(self):
        """
        Everything that needs to be saved in order to resuscitate this agent
        :return: A dictionary with actor critic network parameters
        """
        return {'actor_critic': self.algorithm.actor_critic}

    def train(self, make_env, preprocess):
        # create shared data between agent clones
        rollout = {
            'states': torch.empty(
                self.nb_threads,
                self.nb_rollout_steps+1,
                *self.state_shape).to(self.device).share_memory_(),
            'actions': torch.empty(
                self.nb_threads,
                self.nb_rollout_steps,
                dtype=torch.long).to(self.device).share_memory_(),
            'rewards': torch.empty(
                self.nb_threads,
                self.nb_rollout_steps).to(self.device).share_memory_(),
            'dones': torch.empty(
                self.nb_threads,
                self.nb_rollout_steps).to(self.device).share_memory_(),
            'test_reward': torch.empty(self.max_train_steps//self.test_freq).share_memory_()
        }
        # stopqs and startqs tell when the actors should collect rollouts
        stopqs = []
        startqs = []
        # make the policy available to all processes
        self.algorithm.actor_critic.share_memory()
        procs = []
        # clones = []
        for t in range(self.nb_threads):
            startq = mp.Queue(1)
            startqs.append(startq)
            stopq = mp.Queue(1)
            stopqs.append(stopq)
            c = self.Clone(
                t, preprocess, self.policy, self.nb_rollout_steps, self.k,
                self.device, rollout)
            # clones.append(c)
            proc = mp.Process(target=c.run, args=(self.algorithm.pi, make_env(), startq, stopq))
            procs.append(proc)
            proc.start()
        # have a thread for testing
        test_startq = mp.Queue(1)
        test_stopq = mp.Queue(1)
        test_clone = self.Clone(
                t+1, preprocess, self.policy, self.nb_rollout_steps, self.k,
                self.device, rollout)
        test_proc = mp.Process(
            target=test_clone.test_run,
            args=(self.algorithm.pi, make_env(), test_startq, test_stopq))
        test_proc.start()

        # train
        step = 0
        while step < self.max_train_steps:
            # start collecting data
            for start in startqs:
                start.put(1)
            # wait for the agents to finish getting data
            for stop in stopqs:
                stop.get()
            # for c in clones:
            #     c.run(self.al
            # update!
            metrics = {}
            self.algorithm.update((
                rollout['states'][:,:-1],
                rollout['actions'],
                rollout['rewards'],
                rollout['dones']), rollout['states'][:,-1], metrics)
            step += 1
            # test
            if step % self.test_freq == 0:
                test_startq.put(step // self.test_freq)
                test_stopq.get()
                metrics['test_reward'] = rollout['test_reward'][step // self.test_freq]
                # test_clone.test_run(self.algorithm.pi, testq)
            # anneal algorithm parameters
            self.algorithm.anneal(step, self.max_train_steps)
            # callbacks
            for callback in self.callbacks:
                callback.on_step_end(step, metrics)

        # end by shutting down processes
        for p in procs:
            p.terminate()
        test_proc.terminate()

