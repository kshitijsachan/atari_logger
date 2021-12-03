import envlogger
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
    for episode, metadata in reader.read('/home/ksachan/log/MontezumaRevengeNoFrameskip-v4/1'):
        print(len(episode))

