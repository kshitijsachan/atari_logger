import gym, ale_py, os, ipdb, argparse, pygame, pickle, time, envlogger
import numpy as np

from timer import Timer
from bsuite.utils import gym_wrapper
from collections import deque
from dataclasses import dataclass, field, InitVar
from typing import List, Tuple
from enum import Enum
from gym import logger
from pygame.locals import VIDEORESIZE
from envlogger.backends import schedulers

class Pause(Enum):
    MANUAL = 0
    IDLE = 1

class State(Enum):
    RUNNING = 0 
    MANUAL_PAUSE = 1
    IDLE_PAUSE = 2 
    WAIT_FOR_RESET = 3
    QUIT = 4

class FPSTracker:
    num_frames = 10000
    observed_iter_times = []
    zoom=3

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
            smooth_fps = np.convolve(observed_fps, np.ones(5)/5, mode='valid')
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


class Play:
    def __init__(self, game_name, user_id, base_logging_dir="/home/ksachan/log", should_log=True, fps=60, zoom=3, idle_threshold=1000, dump_frequency=3000):
        self.fps = fps
        self.idle_threshold = idle_threshold
        self.should_log = should_log
        self.dump_frequency = dump_frequency

        # set up env
        gym_env = self._make_env(game_name)
        self.env = gym_wrapper.DMEnvFromGym(gym_env)

        # pygame display vars
        pygame.font.init()
        rendered = gym_env.render(mode="rgb_array")
        self.video_size = int(rendered.shape[0] * zoom), int(rendered.shape[1] * zoom)
        self.screen = pygame.display.set_mode(self.video_size)
        self.state = State.RUNNING

        # reset to end of previous run/set up logging dir
        env_name = self.env.gym_env.unwrapped.spec.id
        self.logging_dir = os.path.join(base_logging_dir, env_name, str(user_id))
        os.makedirs(self.logging_dir, exist_ok=True)

        # pygame keyboard vars
        self.pressed_keys = []
        if hasattr(gym_env, "get_keys_to_action"):
            keys_to_action = gym_env.get_keys_to_action()
        elif hasattr(gym_env.unwrapped, "get_keys_to_action"):
            keys_to_action = gym_env.unwrapped.get_keys_to_action()
        else:
            assert False, f"{env.spec.id} does not have explicit key to action mapping, specify one manually"
        self.keys_to_action = keys_to_action
        self.action_keys = set(sum(map(list, keys_to_action.keys()), []))

        # track idleness
        self.pauses = [] # (most recently seen frame number before pause begins, pause type) List
        self.frame_number = 0
        self.num_consecutive_noops = 0

        # track starting from checkpoint
        self.checkpoint_start = False
        self.backup_file_name = 'backup.pkl'

    def _make_env(self, game_name):
        game_to_env = { 
                "montezuma_revenge" : "MontezumaRevengeNoFrameskip-v4",
                "pitfall" : "PitfallNoFrameskip-v4",
                "venture" : "VentureNoFrameskip-v4"
                }
        try:
            env_name = game_to_env[game_name]
        except KeyError:
            parser.error(f"Unsupported game name: {game_name}\nMust be one of: {list(game_to_env.keys())}")
        return gym.make(env_name)

    def play(self):
        def step_fn(timestep, action, env):
            if self.checkpoint_start:
                return {'checkpoint_start' : True}

            ram_state = env.gym_env.ale.getRAM()
            return {'ram' : ram_state }

        episode_fn = lambda timestep, action, env : self.pauses

        clock = pygame.time.Clock()
        if self.should_log:
            with envlogger.EnvLogger(self.env, 
                                    data_directory=self.logging_dir, 
                                    step_fn=step_fn, 
                                    episode_fn=episode_fn, 
                                    flush_scheduler=schedulers.n_step_scheduler(self.dump_frequency)) as self.env: 
                self._load_backup()
                while self.state != State.QUIT:
                    # with FPSTracker():
                    self._handle_events()
                    self._take_action()
                    self._update_screen()
                    clock.tick(self.fps)
        else:
            while self.state != State.QUIT:
                # with FPSTracker():
                self._handle_events()
                self._take_action()
                self._update_screen()
                clock.tick(self.fps)
        pygame.quit()
        self._write_backup()

    def _write_backup(self):
        with open(os.path.join(self.logging_dir, self.backup_file_name), 'wb') as f:
            last_state = self.env.gym_env.ale.cloneState()
            pickle.dump((last_state, self.pauses, self.frame_number), f)

    def _load_backup(self):
        try:
            with open(os.path.join(self.logging_dir, self.backup_file_name), 'rb') as f:
                last_state, self.pauses, self.frame_number = pickle.load(f)
            # need to call `reset` (but tell dm logger to ignore that reset) before loading checkpoint
            self.checkpoint_start = True
            self.env.reset()
            self.checkpoint_start = False
            self.env.gym_env.ale.restoreState(last_state)
        except FileNotFoundError:
            pass

    def _handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.state = State.QUIT
            elif self.state == State.MANUAL_PAUSE and event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                self.state = State.RUNNING
            elif self.state == State.IDLE_PAUSE and event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                self.state = State.RUNNING
                self.num_consecutive_noops = 0
            elif self.state == State.WAIT_FOR_RESET and event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                self.state = State.RUNNING
                self.frame_number = 0
                self.pauses = []
                self.env.reset()
            elif self.state == State.RUNNING:
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        self.state = State.QUIT
                    elif event.key == pygame.K_p:
                        self.state = State.MANUAL_PAUSE
                        self.pressed_keys = []
                        self._log_pause(Pause.MANUAL)
                    elif event.key in self.action_keys:
                        self.pressed_keys.append(event.key)
                elif event.type == pygame.KEYUP and event.key in self.action_keys:
                    self.pressed_keys.remove(event.key)

    def _take_action(self):
        if self.state == State.RUNNING:
            action = self.keys_to_action.get(tuple(sorted(self.pressed_keys)), 0)
            timestep = self.env.step(action)
            if timestep.last():
                self.state = State.WAIT_FOR_RESET
            elif action == 0:
                self.num_consecutive_noops += 1
                if self.num_consecutive_noops == self.idle_threshold:
                    self.state = State.IDLE_PAUSE
                    self._log_pause(Pause.IDLE)
            self.frame_number += 1

    def _update_screen(self):
        if self.state == State.RUNNING:
            arr = self.env.gym_env.render(mode="rgb_array")
            # arr_scale = 255.0 / 236
            # arr = arr * arr_scale
            arr = arr.swapaxes(0, 1)
            pyg_img = pygame.surfarray.make_surface(arr)
            pygame.transform.scale(pyg_img, self.video_size, self.screen)
        elif self.state == State.MANUAL_PAUSE:
            header = pygame.font.SysFont('Consolas', 35).render('Paused', True, pygame.color.Color('Green'))
            subtext = pygame.font.SysFont('Consolas', 20).render('Press [r] to resume', True, pygame.color.Color('Green'))
            self.screen.blit(header, (10, 5))
            self.screen.blit(subtext, (10, 40))
        elif self.state == State.IDLE_PAUSE:
            header = pygame.font.SysFont('Consolas', 35).render('Are you still playing?', True, pygame.color.Color('Green'))
            subtext = pygame.font.SysFont('Consolas', 20).render('Press [r] to resume', True, pygame.color.Color('Green'))
            self.screen.blit(header, (10, 5))
            self.screen.blit(subtext, (10, 40))
        elif self.state == State.WAIT_FOR_RESET:
            header = pygame.font.SysFont('Consolas', 35).render('Game Over!', True, pygame.color.Color('Green'))
            subtext = pygame.font.SysFont('Consolas', 20).render('Press [r] to play again', True, pygame.color.Color('Green'))
            self.screen.blit(header, (10, 5))
            self.screen.blit(subtext, (10, 40))
        pygame.display.flip()

    def _log_pause(self, pause_type):
        # need to store `value` field of enum so it can be encoded by dm logger
        self.pauses.append((self.frame_number, pause_type.value))
    

def is_valid_file(parser, arg):
    if not os.path.exists(arg):
        parser.error(f"The file {arg} does not exist")
    else:
        return arg



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--game_name", type=str)
    parser.add_argument("--user_id", type=int)
    parser.add_argument("--log_folder", type=str)
    parser.add_argument("--no_log", action='store_true')
    args = parser.parse_args()

    controller = Play(game_name=args.game_name, user_id=args.user_id, base_logging_dir=args.log_folder, should_log=not args.no_log, zoom=3)
    controller.play()
