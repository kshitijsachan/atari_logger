from controller import Controller
import os

user = os.environ.get("USERNAME").strip()
CONFIG_FILE = f"/mnt/c/Users/{user}/Desktop/game_config.txt"


def parse_args():
    try:
        with open(CONFIG_FILE) as f:
            lines = f.readlines()
    except FileNotFoundError:
        raise ValueError("config file is not provided")

    if len(lines) != 2:
        raise ValueError("config file does not have exactly 2 lines")

    user_id = int(lines[0])
    game_name = lines[1]
    logging_dir = '/home/ksachan/log'
    return game_name, user_id, logging_dir


if __name__ == "__main__":
    controller = Controller(*parse_args())
    controller.play()
