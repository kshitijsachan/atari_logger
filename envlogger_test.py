import gym, envlogger, ipdb, random, time
import numpy as np

from bsuite.utils import gym_wrapper


if __name__ == "__main__":
    datadir = "/home/ksachan/atari_logger/envlog"
    
    pauses = [1, 2, 3]

    def step_fn(timestep, action, env):
        step_type, r, gamma, image_state = timestep
        ram_state = env.gym_env.ale.getRAM()
        
        return {'ram' : ram_state, 'pauses' : pauses }

    env = gym.make("ALE/MontezumaRevenge-v5")
    env = gym_wrapper.DMEnvFromGym(env)
    with envlogger.EnvLogger(env, data_directory=datadir, step_fn=step_fn) as env:
        for i in range(20):
            timestep = env.step(random.choice(range(17)))
    with envlogger.EnvLogger(env, data_directory=datadir, step_fn=step_fn) as env:
        for i in range(20):
            timestep = env.step(random.choice(range(17)))

    with envlogger.reader.Reader(data_directory=datadir) as r:
        for i, episode in enumerate(r.episodes):
            for j, step in enumerate(episode):
                ipdb.set_trace()

