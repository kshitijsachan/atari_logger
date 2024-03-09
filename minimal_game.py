# %%
import base64
import logging
import tkinter as tk
from io import BytesIO

import anthropic
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


def get_claude_input(buffer, client):
    messages = [{"type": "text", "text": long_prompt}]
    for i, (prompt, base64_img) in enumerate(
        zip(
            ["\nHere's the first image: ", "junmk", "\nHere's the second image: "],
            buffer.get(),
        )
    ):
        if i != 1:
            messages.append({"type": "text", "text": prompt})
            messages.append(
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": base64_img,
                    },
                }
            )
    messages.append(
        {
            "type": "text",
            "text": "\nAgain, output 'up' or 'down' to move the right-side paddle.",
        }
    )
    print_messages(messages)
    response = client.messages.create(
        model="research-wren-c-s900-200k",
        max_tokens=1024,
        messages=[{"role": "user", "content": messages}],
    )
    print(response)
    action = response.content[0].text
    if "none" in action:
        return (None,)
    if "up" in action:
        return (100,)
    elif "down" in action:
        return (97,)
    return (None,)


# Read the long prompt from a file
with open("pong.txt", "r") as file:
    long_prompt = file.read()


class Queue:
    def __init__(self, max_size):
        self.max_size = max_size
        self.queue = []

    def push(self, item):
        if len(self.queue) == self.max_size:
            self.queue.pop(0)
        self.queue.append(item)

    def get(self):
        return self.queue


def pil_image_to_base64(image):
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    image_data = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return image_data


def print_messages(messages):
    for message in messages:
        if message["type"] == "text":
            print(message["text"])
        elif message["type"] == "image":
            display_base64_image(message["source"]["data"])
        else:
            print("Unknown message type")


# %%
def main(game_name, input_mode):
    client = anthropic.Anthropic(api_key="KEY HERE")
    buffer = Queue(3)
    env = gym.make(game_name, render_mode="rgb_array")
    env.reset()

    # Create a window to display the game
    window = tk.Tk()
    window.title("Montezuma's Revenge")

    # Create a label to hold the game image
    img_label = tk.Label(window)
    img_label.pack()

    action_mapping = env.get_keys_to_action()
    print(action_mapping)

    try:
        for i in range(200):
            # Render the current game state as an image
            img_arr = env.render()
            img_arr = repeat_upsample(img_arr, 10, 10)
            img = Image.fromarray(img_arr)

            # Update the image in the window
            img_tk = ImageTk.PhotoImage(img)
            img_label.configure(image=img_tk)
            img_label.image = img_tk
            window.update()

            buffer.push(pil_image_to_base64(img))

            # Get user input or query the Anthropic API for the action
            if input_mode == "keyboard":
                action = get_user_input(window)
            elif input_mode == "api":
                if i < 44:
                    action = (None,)
                else:
                    action = get_claude_input(buffer, client)
            if action is None:
                break  # User pressed "q" to quit

            print(i, "Action:", action)
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

    finally:
        env.close()
        window.destroy()


# %%
buffer = main("ALE/Pong-v5", "api")

# %%
DisplayImage(data=buffer.get()[0])
# %%

main("ALE/Pong-v5", "keyboard")

# %%
import base64
from io import BytesIO

from PIL import Image

# Base64-encoded image data
base64_image = buffer.get()[0]

# Decode the base64 image data

display_base64_image(base64_image)
# # Open the image using Pillow
# image = Image.open(image_stream)

# # Display the image
# image.show()
# %%
