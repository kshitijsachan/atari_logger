import ipdb
import time
from collections import deque

import numpy as np


class FPSTracker:
    num_frames = 10000
    observed_iter_times = []
    zoom = 3

    def __init__(self):
        pass

    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, *args):
        import matplotlib.pyplot as plt
        end = time.time()
        self.observed_iter_times.append(end - self.start)
        if len(self.observed_iter_times) == self.num_frames:
            observed_fps = 1 / np.array(self.observed_iter_times)
            plt.plot(range(len(observed_fps)), observed_fps)
            plt.title(f"zoom={self.zoom}x")
            plt.savefig(f"{self.zoom}x_zoom.png")
            plt.close('all')
            smooth_fps = np.convolve(observed_fps, np.ones(5) / 5, mode='valid')
            plt.plot(range(len(smooth_fps)), smooth_fps)
            plt.title(f"zoom={self.zoom}x_smoothwindow=5")
            plt.savefig(f"{self.zoom}x_zoom_smooth.png")
            plt.close('all')
            ipdb.set_trace()


class MovingAverage:
    def __init__(self, horizon):
        self.d = deque()
        self.curr_sum = 0
        self.horizon = horizon

    def update(self, val):
        self.curr_sum += val
        self.d.append(val)
        if len(self.d) > self.horizon:
            self.curr_sum -= self.d.popleft()
            return self.curr_sum / self.horizon
