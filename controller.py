from enum import Enum

import ipdb
import argparse
import envlogger
import gym
import os
import pickle
import pygame
import time
from bsuite.utils import gym_wrapper
from envlogger.backends import schedulers
from fps_tracker import FPSTracker


class Pause(Enum):
    MANUAL = 0
    IDLE = 1


class State(Enum):
    RUNNING = 0
    MANUAL_PAUSE = 1
    IDLE_PAUSE = 2
    WAIT_FOR_RESET = 3
    QUIT = 4


class Controller:
    def __init__(self, game_name, user_id, base_logging_dir, should_log=True,
                 fps=60, zoom=3, idle_threshold=1000, dump_frequency=3000):
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
            assert False, f"{gym_env.spec.id} does not have explicit key to action mapping, specify one manually"
        self.keys_to_action = keys_to_action
        self.action_keys = set(sum(map(list, keys_to_action.keys()), []))

        # track idleness
        self.state = State.RUNNING
        self.pauses = []  # (most recently seen frame number before pause begins, pause type) List
        self.frame_number = 0
        self.num_consecutive_noops = 0

        # track starting from checkpoint
        self.checkpoint_start = False
        self.backup_file_name = 'backup.pkl'
        self.time_offset = 0
        self.start_time = time.time()

        # stop blitting repeatedly on pause
        self.was_running = True 

    @staticmethod
    def _make_env(game_name):
        game_to_env = {
            "montezuma_revenge": "MontezumaRevengeNoFrameskip-v4",
            "pitfall": "PitfallNoFrameskip-v4",
            "venture": "VentureNoFrameskip-v4"
        }
        try:
            env_name = game_to_env[game_name]
            return gym.make(env_name)
        except KeyError:
            parser.error(f"Unsupported game name: {game_name}\nMust be one of: {list(game_to_env.keys())}")

    def play(self):
        """Main function that turns on logging, runs game, and saves checkpoint"""

        def step_fn(timestep, action, env):
            if self.checkpoint_start:
                return {'checkpoint_start': True}
            ram_state = env.gym_env.ale.getRAM()
            curr_time = time.time()
            time_played = curr_time - self.start_time + self.time_offset
            return {'ram': ram_state, 'time': time_played}

        def episode_fn(timestep, action, env):
            return self.pauses

        scheduler = schedulers.n_step_scheduler(self.dump_frequency)
        if self.should_log:
            with envlogger.EnvLogger(self.env,
                                     data_directory=self.logging_dir,
                                     step_fn=step_fn,
                                     episode_fn=episode_fn,
                                     flush_scheduler=scheduler) as self.env:
                self._load_backup()
                self._run_loop()
                self._write_backup()
        else:
            self._run_loop()
        pygame.quit()

    def _run_loop(self):
        """Handles user actions and updates screen until user closes window"""
        clock = pygame.time.Clock()
        while self.state != State.QUIT:
            # with FPSTracker():
            self._handle_events()
            self._take_action()
            self._update_screen()
            clock.tick(self.fps)

    def _handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.state = State.QUIT
            # manually override any game resizes
            elif event.type == pygame.VIDEORESIZE:
                self.screen = pygame.display.set_mode(self.video_size)
            elif self.state == State.MANUAL_PAUSE and event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                self._state_transition(State.RUNNING)
            elif self.state == State.IDLE_PAUSE and event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                self._state_transition(State.RUNNING)
                self.num_consecutive_noops = 0
            elif self.state == State.WAIT_FOR_RESET and event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                self._state_transition(State.RUNNING)
                self.frame_number = 0
                self.pauses = []
                self.pressed_keys = []
                self.env.reset()
            elif self.state == State.RUNNING:
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        self._state_transition(State.QUIT)
                    elif event.key == pygame.K_p:
                        self._state_transition(State.MANUAL_PAUSE)
                        self.pressed_keys = []
                    elif event.key in self.action_keys:
                        self.pressed_keys.append(event.key)
                elif event.type == pygame.KEYUP and event.key in self.action_keys:
                    self.pressed_keys.remove(event.key)

    def _take_action(self):
        if self.state == State.RUNNING:
            action = self.keys_to_action.get(tuple(sorted(self.pressed_keys)), 0)
            timestep = self.env.step(action)
            if timestep.last():
                self._state_transition(State.WAIT_FOR_RESET)
            elif action == 0:
                self.num_consecutive_noops += 1
                if self.num_consecutive_noops == self.idle_threshold:
                    self._state_transition(State.IDLE_PAUSE)
                    self._log_pause(Pause.IDLE)
            else:
                self.num_consecutive_noops = 0
            self.frame_number += 1

    def _update_screen(self):
        if self.state == State.RUNNING:
            arr = self.env.gym_env.render(mode="rgb_array")
            arr = arr.swapaxes(0, 1)
            pyg_img = pygame.surfarray.make_surface(arr)
            pygame.transform.scale(pyg_img, self.video_size, self.screen)

            # display time played so far
            time_played_sec = int(time.time() - self.start_time + self.time_offset)
            hours = time_played_sec // 3600
            minutes = (time_played_sec // 60) % 60
            sec = time_played_sec % 60
            time_played_str = "Time played: %02d:%02d:%02d" % (hours, minutes, sec) 
            bottom_header = pygame.font.SysFont('Consolas', 20).render(time_played_str, True, pygame.color.Color('Green'))
            self.screen.blit(bottom_header, (10, self.video_size[1] - 26))
        elif self.state == State.MANUAL_PAUSE:
            header = pygame.font.SysFont('Consolas', 30).render('Paused', True, pygame.color.Color('Green'))
            subtext = pygame.font.SysFont('Consolas', 20).render('Press [r] to resume', True, pygame.color.Color('Green'))
            self.screen.blit(header, (10, 5))
            self.screen.blit(subtext, (10, 40))
        elif self.state == State.IDLE_PAUSE:
            header = pygame.font.SysFont('Consolas', 30).render('Still playing?', True, pygame.color.Color('Green'))
            subtext = pygame.font.SysFont('Consolas', 20).render('Press [r] to resume', True, pygame.color.Color('Green'))
            self.screen.blit(header, (10, 5))
            self.screen.blit(subtext, (10, 40))
        elif self.state == State.WAIT_FOR_RESET:
            header = pygame.font.SysFont('Consolas', 30).render('Game Over!', True, pygame.color.Color('Green'))
            subtext = pygame.font.SysFont('Consolas', 20).render('Press [r] to play again', True, pygame.color.Color('Green'))
            self.screen.blit(header, (10, 5))
            self.screen.blit(subtext, (10, 40))

        # update screen (but not if game is not running because there is a weird bug that repeatedly prints text)
        if self.state == State.RUNNING:
            pygame.display.flip()
            self.was_running = True
        elif self.was_running:
            self.was_running = False
            pygame.display.flip()

    def _write_backup(self):
        """Save the last state, pauses during episode, and current frame number to file"""
        with open(os.path.join(self.logging_dir, self.backup_file_name), 'wb') as f:
            last_state = self.env.gym_env.ale.cloneState()
            total_time_played = self.time_offset
            if hasattr(self, 'start_time'):
                unfinished_episode_time_played = time.time() - self.start_time
                total_time_played += unfinished_episode_time_played
            pickle.dump((last_state, self.pauses, self.frame_number, total_time_played), f)

    def _load_backup(self):
        """Load checkpoint if it exists"""
        try:
            with open(os.path.join(self.logging_dir, self.backup_file_name), 'rb') as f:
                last_state, self.pauses, self.frame_number, self.time_offset = pickle.load(f)
            # need to call `reset` (but tell dm logger to ignore that reset) before loading checkpoint
            self.checkpoint_start = True
            self.env.reset()
            self.checkpoint_start = False
            self.env.gym_env.ale.restoreState(last_state)
        except FileNotFoundError:
            pass

    def _state_transition(self, new_state):
        """Transition to new internal state and handle timing and pause logging logic"""
        # handle timer
        curr_time = time.time()
        if new_state == State.RUNNING:
            self.start_time = curr_time
        else:
            self.time_offset += curr_time - self.start_time
            del self.start_time

        # log pauses
        if new_state == State.MANUAL_PAUSE:
            self._log_pause(Pause.MANUAL)
        elif new_state == State.IDLE_PAUSE:
            self._log_pause(Pause.IDLE)
        self.state = new_state

    def _log_pause(self, pause_type):
        # need to store `value` field of enum so it can be encoded by dm logger
        self.pauses.append((self.frame_number, pause_type.value))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--game_name", type=str)
    parser.add_argument("--user_id", type=int)
    parser.add_argument("--log_folder", type=str)
    parser.add_argument("--no_log", action='store_true')
    args = parser.parse_args()

    controller = Controller(game_name=args.game_name, user_id=args.user_id, base_logging_dir=args.log_folder, should_log=not args.no_log)
    controller.play()
