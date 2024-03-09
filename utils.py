# %%
import base64
from io import BytesIO
from itertools import islice, tee

import cv2
import numpy as np
from IPython.display import Image as IPythonImage
from IPython.display import display
from PIL import Image


def sliding_window(iterable, window_len: int, interwindow_step: int = 1):
    return [
        list(
            islice(
                iterable, start, start + window_len * interwindow_step, interwindow_step
            )
        )
        for start in range(len(iterable) - (window_len * interwindow_step) + 1)
    ]


def expand_frame(frame: np.ndarray, expansion_factor: int):
    assert frame.ndim == 3
    assert expansion_factor > 0
    return np.repeat(
        np.repeat(frame, expansion_factor, axis=0), expansion_factor, axis=1
    )


def render_image(image_array: np.ndarray) -> None:
    assert image_array.ndim == 3
    image = Image.fromarray(image_array)
    image.show()


def render_video(video_array, frame_ixs=None):
    num_frames, height, width = video_array.shape[:3]
    frame_ixs = frame_ixs or list(range(num_frames))
    current_ix = 0

    while True:
        frame = video_array[current_ix]
        frame_with_text = cv2.putText(
            frame.copy(),
            f"#{frame_ixs[current_ix]}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )
        cv2.imshow("Video", frame_with_text)

        key = cv2.waitKey(0) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("a") or key == 81:  # Left arrow key
            current_ix = (current_ix - 1) % num_frames
        elif key == ord("d") or key == 83:  # Right arrow key
            current_ix = (current_ix + 1) % num_frames

    cv2.destroyAllWindows()


def display_base64_image(base64_image):
    image_data = base64.b64decode(base64_image)
    image_stream = BytesIO(image_data)
    display(IPythonImage(image_stream.getvalue()))


def print_messages(messages):
    for message in messages:
        if message["type"] == "text":
            print(message["text"])
        elif message["type"] == "image":
            display_base64_image(message["source"]["data"])
        else:
            print("Unknown message type")


expand_frame(np.random.rand(10, 10, 3), 2).shape

# %%
