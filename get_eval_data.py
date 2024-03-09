# %%
import base64
import logging
import tkinter as tk
from io import BytesIO

import gym
import numpy as np
from IPython.display import Image as PythonImage
from IPython.display import display
from PIL import Image, ImageTk


def repeat_upsample(rgb_array, width_repeats=1, height_repeats=1):
    assert width_repeats > 0
    assert height_repeats > 0
    return np.repeat(
        np.repeat(rgb_array, height_repeats, axis=0), width_repeats, axis=1
    )


def display_base64_image(base64_image):
    image_data = base64.b64decode(base64_image)
    image_stream = BytesIO(image_data)
    display(PythonImage(image_stream.getvalue()))


def map_keys_to_action(keys):
    action = set()
    for key in keys:
        if key == "space":
            action.add(32)
        elif key == "w":
            action.add(119)
        elif key == "a":
            action.add(97)
        elif key == "s":
            action.add(115)
        elif key == "d":
            action.add(100)
    return tuple(sorted(action))


def get_user_input(window):
    valid_keys = {"w", "a", "s", "d", "space", "q", "Return"}
    keys_pressed = set()

    def on_key_press(event):
        if event.keysym in valid_keys:
            keys_pressed.add(event.keysym)

    window.bind("<KeyPress>", on_key_press)

    while True:
        window.update()
        if "q" in keys_pressed:
            break
        if "Return" in keys_pressed:
            keys_pressed.discard("Return")
            if len(keys_pressed) == 0:
                return (None,)
            action = map_keys_to_action(keys_pressed)
            keys_pressed.clear()
            return action

    return None


# %%
game_name = "ALE/Pong-v5"
env = gym.make(game_name, render_mode="rgb_array")
env.reset()

# Create a window to display the game
window = tk.Tk()
window.title(game_name)

# Create a label to hold the game image
img_label = tk.Label(window)
img_label.pack()

action_mapping = env.get_keys_to_action()  # type: ignore
print(action_mapping)
trajectory = []

while True:
    # Render the current game state as an image
    img_arr = env.render()
    trajectory.append(img_arr)
    img_arr = repeat_upsample(img_arr, 2, 2)
    img = Image.fromarray(img_arr)

    # Update the image in the window
    img_tk = ImageTk.PhotoImage(img)
    img_label.configure(image=img_tk)
    img_label.image = img_tk  # type: ignore
    window.update()

    # Get user input or query the Anthropic API for the action
    action = get_user_input(window)
    if action is None:
        break
    print(action)

    if action not in action_mapping:
        logging.warning(f"Invalid action: {action}")
        print("Invalid action")
        continue
    action_idx = action_mapping.get(
        action, 0
    )  # Default to 0 if the action is not found

    # Take the action and get the next state
    obs, reward, done, truncated, info = env.step(action_idx)
    if done:
        print("Game over!")
        break


# %%
trajectory = np.array(trajectory)
np.save("pong_trajectory.npy", trajectory)
