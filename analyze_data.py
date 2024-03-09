# %%
import numpy as np

from utils import render_video

video = np.load("pong_trajectory.npy")
# %%
render_video(video)
# %%
