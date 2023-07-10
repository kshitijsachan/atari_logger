# Research Guidelines
**Please read [this Google doc](https://docs.google.com/document/d/1kmVjI3rQDpYbZ9f0FFH9Xrlej-UiO0njGLf4rNWkiOE/edit?usp=sharing) for instructions on participating in the study!**

# Atari Logger Overview
This library enables you to run an emulator of Atari 2600 games. It logs the details of each frame to a file (keystrokes, game state, score, time played since the app was first opened, etc.). This library was developed to measure how quickly humans learn Atari games. My research group (Brown's Intelligent Robotics Lab) has been developing new skill discovery reinforcement learning algorithms, and we wanted to compare our sample efficiency to some human baselines.

# How to use this?
If you are a participant in the study, you'll recieve a Thinkpad. First, run `configure_laptop.sh` to set up your environment correctly and then run `play.py` whenever you want to play the game. The following screen should pop up:
![montezumas_revenge_screenshot](https://github.com/kshitijsachan/atari_logger/assets/29048943/03e27741-9561-425c-b5df-9f1059c5066f)

`reader.py` enables me to read the data that has been logged and graph your score over time, track when you've paused, etc.

## Keyboard controls
To play the games, you need to use the W, A, S, D, and Space keys. You can press 'P' at any time to pause the game and 'R' to resume (screenshot below). If you don't press any keys for 1 minute, the game will automatically pause (this is important because we want to measure how quickly you improve at the game, so we want to only track when you are actually playing).

You can resize the game window, but making it larger will slow down the game. I would advise against making the game full screen because it will likely lead to lag, and it's crucial that the data we collect is at around 50 frames per second.

The total time you've been playing for is displayed at the bottom of the screen. Even after you close the game and open it again, that timer will not reset. You can close the window in a middle of a game and it will save your state.
![montezumas_revenge_screenshot_paused](https://github.com/kshitijsachan/atari_logger/assets/29048943/ec9c76be-51eb-4eea-b872-f2cc573049ef)


# How does this work?
The Atari backend is implemented via the [OpenAI Gym environment](https://github.com/openai/gym), and the screen is rendered using (pygame)[https://www.pygame.org/news]. I've implemented a wrapper that reads in keystrokes and converts them into RL environment actions. The wrapper also writes your game state to a file at the end of each episode (every time you die).
