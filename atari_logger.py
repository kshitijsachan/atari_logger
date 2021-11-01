import gym, ale_py, os, ipdb, argparse, pygame, pickle
import numpy as np

from timer import Timer

from dataclasses import dataclass, field, InitVar
from typing import List, Tuple
from enum import Enum
from gym import logger
from pygame.locals import VIDEORESIZE

class Pause(Enum):
    MANUAL = 0
    IDLE = 1

class State(Enum):
    RUNNING = 0 
    MANUAL_PAUSE = 1
    IDLE_PAUSE = 2 
    WAIT_FOR_RESET = 3
    QUIT = 4

class LoggerEnv(gym.Wrapper):
    @dataclass
    class EpisodeData:
        init_ram_state: InitVar[np.ndarray]
        episode_number: int = 0
        absolute_frame_number: int = 0
        episode_frame_number: int = 0
        RAM_states: List[np.ndarray] = field(default_factory=list)
        actions: List[int] = field(default_factory=list)
        rewards: List[float] = field(default_factory=list)
        pauses: List[Tuple[int, Pause]] = field(default_factory=list)

        def __post_init__(self, init_ram_state):
            self.RAM_states.append(init_ram_state)

        def step(self, ram_state, a, r):
            self.absolute_frame_number += 1
            self.episode_frame_number += 1
            self.RAM_states.append(ram_state)
            self.actions.append(a)
            self.rewards.append(r)

        def reset(self, init_ram_state):
            self.absolute_frame_number += 1
            self.episode_frame_number = 1
            self.episode_number += 1
            self.RAM_states = [init_ram_state]
            self.actions = []
            self.rewards = []
            self.pauses = []

        def log_pause(self, pause_type):
            self.pauses.append((self.episode_frame_number, pause_type))

        def get_pickle_data(self):
            absolute_first_frame_of_episode = self.absolute_frame_number -self.episode_frame_number
            return (
                self.episode_number, 
                absolute_first_frame_of_episode,
                self.RAM_states,
                self.actions,
                self.rewards,
                self.pauses
            )

    def __init__(self, env, log_folder: str, user_id : int):
        super().__init__(env)
        try:
            self.ale_env = env.ale
        except Exception:
            raise ValueError("LoggerWrapper can only be used on envs of type AtariEnv")

        # set up datadirs
        DATA_FOLDER_NAME = "data"
        BACKUP_FOLDER_NAME = ".backup"
        game_name = env.unwrapped.spec.id
        folder_extension = os.path.join(game_name, str(user_id))
        reset_pickle_dir = os.path.join(log_folder, BACKUP_FOLDER_NAME, folder_extension)
        self.data_pickle_file = os.path.join(log_folder, DATA_FOLDER_NAME, folder_extension, "dataset.pkl")
        self.image_folder = os.path.join(log_folder, DATA_FOLDER_NAME, folder_extension, "images/")
        self.reset_pickle_file = os.path.join(reset_pickle_dir, "reset.pkl")
        os.makedirs(self.image_folder, exist_ok=True)
        os.makedirs(reset_pickle_dir, exist_ok=True)

        # Reset env and initialize logging variables, either from previous 
        # interrupted run or fresh start
        env.reset()
        try:
            with open(self.reset_pickle_file, 'rb') as f:
                self.log = pickle.load(f)
                for i, ram in enumerate(self.log.RAM_states[-1]):
                    self.ale_env.setRAM(i, ram)
        except FileNotFoundError:
            self.log = self.EpisodeData(self.ale_env.getRAM())

        # track how long player has been idle for
        self.num_noops = 0

    def _save_screen(self):
        png_filepath = os.path.join(self.image_folder,
                f"{self.log.absolute_frame_number}_ep={self.log.episode_number}_frame={self.log.episode_frame_number}.png")
        self.ale_env.saveScreenPNG(png_filepath)

    def step(self, action):
        obs, reward, done, info = super().step(action)

        # Track idleness
        if action == 0:
            self.num_noops += 1
            if self.num_noops == 1000:
                info['idle'] = True
                self.num_noops = 0
        else:
            self.num_noops = 0

        self.log.step(self.ale_env.getRAM(), action, reward)
        self._save_screen()

        # write episode data to pickle file
        if done:
            with open(self.data_pickle_file, 'ab') as f:
                pickle.dump(self.log.get_pickle_data(), f)
        return obs, reward, done, info

    def reset(self, **kwargs):
        observations = super().reset(**kwargs)
        self.log.reset(self.ale_env.getRAM())
        self._save_screen()
        return observations

    def graceful_exit(self):
        with open(self.reset_pickle_file, 'wb') as f:
            pickle.dump(self.log, f)
            print("Exit successful: Dumped log to .backup file")


class Play:
    def __init__(self, env: LoggerEnv, fps=60, zoom=3):
        pygame.font.init()
        self.env = env
        rendered = self.env.render(mode="rgb_array")
        
        self.fps = fps
        self.video_size = int(rendered.shape[1] * zoom), int(rendered.shape[0] * zoom)
        self.screen = pygame.display.set_mode(self.video_size)
        self.state = State.RUNNING
        self.pressed_keys = []

        if hasattr(env, "get_keys_to_action"):
            keys_to_action = env.get_keys_to_action()
        elif hasattr(env.unwrapped, "get_keys_to_action"):
            keys_to_action = env.unwrapped.get_keys_to_action()
        else:
            assert False, f"{env.spec.id} does not have explicit key to action mapping, specify one manually"
        self.keys_to_action = keys_to_action
        self.action_keys = set(sum(map(list, keys_to_action.keys()), []))

    def play(self):
        clock = pygame.time.Clock()
        with Timer('total'):
            while self.state != State.QUIT:
                with Timer('_update_screen'):
                    self._update_screen()
                with Timer('_take_action'):
                    self._take_action()
                with Timer('_handle_events'):
                    self._handle_events()
                clock.tick(self.fps)
        Timer.print_stats()
        pygame.quit()
        self.env.graceful_exit()

    def _handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.state = State.QUIT
            elif event.type == VIDEORESIZE:
                self.video_size = event.size
                # TODO:
                # self.screen = pygame.display.set_mode(video_size)
            elif self.state == State.MANUAL_PAUSE and event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                self.state = State.RUNNING
            elif self.state == State.IDLE_PAUSE and event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                self.state = State.RUNNING
            elif self.state == State.WAIT_FOR_RESET and event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                self.state = State.RUNNING
                self.env.reset()
            elif self.state == State.RUNNING:
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        self.state = State.QUIT
                    elif event.key == pygame.K_p:
                        self.state = State.MANUAL_PAUSE
                        self.pressed_keys = []
                        self.env.log.log_pause(Pause.MANUAL)
                    elif event.key in self.action_keys:
                        self.pressed_keys.append(event.key)
                elif event.type == pygame.KEYUP and event.key in self.action_keys:
                    self.pressed_keys.remove(event.key)

    def _take_action(self):
        if self.state == State.RUNNING:
            action = self.keys_to_action.get(tuple(sorted(self.pressed_keys)), 0)
            obs, rew, env_done, info = self.env.step(action)
            if 'idle' in info:
                self.state = State.IDLE_PAUSE
                self.env.log.log_pause(Pause.IDLE)
            elif env_done:
                self.state = State.WAIT_FOR_RESET

    def _update_screen(self):
        if self.state == State.RUNNING:
            arr = self.env.render(mode="rgb_array")
            arr_min, arr_max = arr.min(), arr.max()
            arr = 255.0 * (arr - arr_min) / (arr_max - arr_min)
            pyg_img = pygame.surfarray.make_surface(arr.swapaxes(0, 1))
            pyg_img = pygame.transform.scale(pyg_img, self.video_size)
            self.screen.blit(pyg_img, (0, 0))
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
    

def is_valid_file(parser, arg):
    if not os.path.exists(arg):
        parser.error(f"The file {arg} does not exist")
    else:
        return arg

def atari_env(game_name):
    game_to_env = { "montezuma_revenge" : "MontezumaRevengeNoFrameskip-v4" }
    try:
        return game_to_env[game_name]
    except KeyError:
        parser.error(f"Unsupported game name: {game_name}\nMust be one of: {list(game_to_env.keys())}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--game_name", type=atari_env)
    parser.add_argument("--user_id", type=int)
    parser.add_argument("--log_folder", type=str)
    args = parser.parse_args()

    env = gym.make(args.game_name)
    env.reset()
    # env = LoggerEnv(env, args.log_folder, args.user_id)

    controller = Play(env)
    controller.play()
