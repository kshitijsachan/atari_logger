import envlogger
import matplotlib.pyplot as plt
import ipdb

from dm_env import StepType

class Reader:
    def _reset_state(self):
        self.cleaned_episode = []
        self.cleaned_metadata = []
        # pause numbers are offset because they are stored relative to 
        # position in that episode, even though the episode may just be
        # a checkpoint, not a complete episode
        self.pause_offset = 0

    def read(self, filepath):
        self._reset_state()
        with envlogger.reader.Reader(data_directory=filepath) as r:
            for episode, metadata in zip(r.episodes, r.episode_metadata()):
                if metadata is not None:
                    self.cleaned_metadata.extend(
                            [(frame_number + self.pause_offset, pause_type) 
                                for (frame_number, pause_type) in metadata])
                for step in episode:
                    # ignore checkpoint dummy restart states
                    if 'checkpoint_start' in step.custom_data:
                        continue

                    # else, store this step and increment offset
                    self.cleaned_episode.append(step)
                    self.pause_offset += 1
                    if step.timestep.step_type == StepType.LAST:
                        yield self.cleaned_episode, self.cleaned_metadata
                        self._reset_state()


if __name__ == "__main__":
    reader = Reader()
    rewards = []
    cnt = 0
    for episode, metadata in reader.read('/home/ksachan/log/MontezumaRevengeNoFrameskip-v4/3'):
        cnt += 1
        reward = 0
        first = True
        for step in episode:
            if first:
                seconds = int(step[2]['time'])
                print_string = f"Episode: {cnt} Time: {seconds // 3600}:{(seconds // 60) % 60}:{seconds % 60}"
                first = False
            if step.timestep.reward is not None:
                reward += step.timestep.reward
        rewards.append(reward)
        print(print_string, f"Reward: {reward}")
    plt.plot(rewards)
    plt.xlim([0, len(rewards)])
    plt.xlabel("episode #")
    plt.ylabel("reward")
    plt.savefig('rewards.png')
