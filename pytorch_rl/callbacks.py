from pytorch_rl.utils import *
import time
import os
import pickle

class Callback(object):
    """abstract class"""

    def on_step_end(self, step, step_log):
        pass

    def on_episode_end(self, ep, episode_log):
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

    def on_step_end(self, step, step_log):
        if step % self.freq == 0:
            to_print = '{}'.format(step)
            for key, value in sorted(step_log.items(), key=lambda x: x[0]):
                if type(value) == AverageMeter:
                    to_print += ", {}: {:.3f}".format(key, value.avg)
            to_print += ", time/iter (ms): {:.3f}".format(1000 * (time.time() - self.start)/self.freq)
            print(to_print)
            self.start = time.time()


class SaveNetworks(Callback):
    def __init__(self, save_dir, freq, network_func):
        self.save_dir = save_dir
        self.freq = freq
        self.networks = network_func

    def on_step_end(self, step, step_log):
        if step % self.freq == 0:
            nets = self.networks()
            for name, net in nets.items():
                path = os.path.join(self.save_dir, name+'_{}.pth'.format(step))
                print("saving {} network to {} path".format(name, path))
                torch.save(net, path)


class SaveMetrics(Callback):
    def __init__(self, save_dir, freq):
        self.save_dir = save_dir
        self.freq = freq
        self.metrics = {}
        self.save_count = 0

    def on_step_end(self, step, step_log):
        for key, value in step_log.items():
            if key in self.metrics.keys():
                self.metrics[key].append((step, value.avg))
            else:
                self.metrics[key] = [(step, value.avg)]
        if step % self.freq == 0:
            self.save_count += 1
            path = os.path.join(self.save_dir, 'metrics_{}.pkl'.format(self.save_count))
            pickle.dump(self.metrics, open(path, 'wb'))
            self.metrics = {}


class TrajectoryDump(Callback):
    def __init__(self, freq, dump_dir):
        self.dump_dir = dump_dir
        self.freq = freq
        self.data = {}
        self.dumps = 0
        if not os.path.exists(dump_dir):
            os.makedirs(dump_dir)

    def on_step_end(self, step, step_log):
        #TODO: need to test this (probably a good idea anyways to check alogorithm)
        for key, value in step_log.items():
            if 'dump/' in key:
                if key in self.data.keys():
                    self.data[key].append(value)
                else:
                    self.data[key] = [value]
        if step % self.freq == 0:
            for key, value in self.data.items():
                self.data[key] = torch.cat([v.unsqueeze(dim=1) for v in value], dim=1)
            outfile = self.dump_dir + str(self.dumps) + '.pt'
            torch.save(self.data, outfile)
            self.dumps +=1
            self.data = {}
