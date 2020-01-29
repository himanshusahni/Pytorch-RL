from pytorch_rl.utils import *
import time
import os

class Callback(object):
    """abstract class"""

    def on_step_end(self, step_log):
        pass

    def on_episode_end(self, episode_log):
        pass


class PlotCallback(Callback):
    def __init__(self, plotter, freq):
        self.plotter = plotter
        self.freq = freq

    def on_step_end(self, step_log):
        if step_log['step'] % self.freq == 0:
            for key, value in step_log.items():
                if type(value) == AverageMeter:
                    self.plotter.plot(key, 'train', key, step_log['step'], value.avg)


class PrintCallback(Callback):
    def __init__(self, freq):
        self.freq = freq
        self.start = time.time()

    def on_step_end(self, step_log):
        if step_log['step'] % self.freq == 0:
            to_print = '{}'.format(step_log['step'])
            for key, value in step_log.items():
                if type(value) == AverageMeter:
                    to_print += ", {}: {:.3f}".format(key, value.avg)
            to_print += ", time/iter (ms): {:.3f}".format(1000 * (time.time() - self.start)/self.freq)
            print(to_print)
            self.start = time.time()


class TrajectoryDump(Callback):
    def __init__(self, freq, dump_dir):
        self.dump_dir = dump_dir
        self.freq = freq
        self.reset()
        self.dumps = 0

    def reset(self):
        """dump the rest of the trajectory and reset"""
        self.states = []
        self.actions = []
        self.rewards = []
        self.dones = []

    def on_step_end(self, step_log):
        #TODO: need to test this (probably a good idea anyways to check alogorithm)
        self.states.append(step_log['states'])
        self.actions.append(step_log['actions'])
        self.rewards.append(step_log['rewards'])
        self.dones.append(step_log['dones'])
        if step_log['step'] % self.freq == 0:
            states = torch.cat([state.unsqueeze(dim=0) for state in self.states]).transpose(0, 1)
            actions = torch.cat([action.unsqueeze(dim=0) for action in self.actions]).transpose(0, 1)
            rewards = torch.cat([reward.unsqueeze(dim=0) for reward in self.rewards]).transpose(0, 1)
            dones = torch.cat([done.unsqueeze(dim=0) for done in self.dones]).transpose(0, 1)
            for t in range(states.size(0)):
                outdir = self.dump_dir + '/' + str(t) + '/'
                if not os.path.exists(outdir):
                    os.makedirs(outdir)
                outfile = outdir + str(self.dumps) + '.pt'
                torch.save({
                    'states': states[t],
                    'actions': actions[t],
                    'rewards': rewards[t],
                    'dones': dones[t],
                }, outfile)
            self.dumps +=1
            self.reset()
