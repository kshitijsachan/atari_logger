# %%
import base64
import dataclasses
import io

import anthropic
import numpy as np
from PIL import Image

from utils import (
    expand_frame,
    print_messages,
    render_image,
    render_video,
    sliding_window,
)

VIDEO = np.load("pong_trajectory.npy")
CLIENT = anthropic.Anthropic(api_key="KEY")


# %%
@dataclasses.dataclass
class AtariLabel:
    ixs: list[int]
    video: np.ndarray
    label: str

    @classmethod
    def construct(cls, ixs: list[int], expansion_factor: int, label: str):
        video = VIDEO[ixs]
        video = np.array([expand_frame(img, expansion_factor) for img in video])
        return cls(ixs, video, label)

    def render(self):
        render_video(self.video, self.ixs)

    def to_b64(self) -> list[str]:
        ans = []
        for img in self.video:
            buffer = io.BytesIO()
            Image.fromarray(img).save(buffer, format="PNG")
            ans.append(base64.b64encode(buffer.getvalue()).decode("utf-8"))  # type: ignore
        return ans


def expand_compressed_labels(
    compressed_labels: list[tuple[int, int, str]],
    n_stack: int,
    skip: int,
    expansion_factor: int,
):
    labels = []
    for start, end, label in compressed_labels:
        all_ixs = list(range(start, end + 1))
        for ixs in sliding_window(all_ixs, n_stack, skip):
            labels.append(AtariLabel.construct(ixs, expansion_factor, label))

    return labels


def make_left_paddle_direction_prompt(label: AtariLabel):
    imgs_content = [
        {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/png",
                "data": image_data,
            },
        }
        for image_data in label.to_b64()
    ]
    content = imgs_content + [
        {
            "type": "text",
            "text": "This is 4 consecutive frames from the Atari game Pong. Tell me if the left paddle is moving up, down, or not at all. Include <up>, <down>, or <none> in your response.",
        }
    ]
    print_messages(content)

    res = CLIENT.messages.create(  # type: ignore
        model="claude-3-opus-20240229",
        max_tokens=256,
        messages=[{"role": "user", "content": content}],
    )
    response = res.content[0].text
    is_correct = f"<{label.label}>" in response
    return is_correct, response


left_paddle_direction_compressed_labels = [
    (16, 30, "down"),
    (31, 50, "up"),
    (50, 70, "down"),
    (70, 93, "up"),
    (93, 121, "down"),
    (121, 150, "none"),
    (150, 165, "up"),
    (165, 184, "down"),
    (211, 240, "up"),
    (240, 261, "down"),
    (262, 283, "up"),
]

puck_vertical_direction_compressed_labels = [
    (16, 30, "down"),
    (30, 48, "up"),
    (48, 60, "down"),
    (60, 69, "down"),
    (69, 88, "up"),
    (94, 120, "down"),
    (121, 150, "flat"),
    (151, 164, "up"),
    (164, 180, "down"),
    (187, 210, "flat"),
]

puck_horizontal_direction_compressed_labels = [
    (16, 30, "left"),
    (30, 48, "right"),
    (48, 60, "right"),
    (60, 69, "left"),
    (69, 88, "left"),
    (94, 120, "right"),
    (121, 150, "left"),
    (151, 164, "right"),
    (164, 180, "right"),
    (187, 210, "left"),
]

right_paddle_direction_compressed_labels = [
    (1, 57, "none"),
    (57, 60, "down"),
    (117, 120, "down"),
    (178, 182, "down"),
]

right_paddle_is_too_blank_compressed_labels = [
    (31, 56, "high"),
    (92, 116, "high"),
    (211, 227, "low"),
]

# %%
all_labels = (
    left_paddle_direction_compressed_labels[:4]
    # + puck_vertical_direction_compressed_labels
    # + puck_horizontal_direction_compressed_labels
    # + right_paddle_direction_compressed_labels
    # + right_paddle_is_too_blank_compressed_labels
)

out = expand_compressed_labels(all_labels, n_stack=4, skip=3, expansion_factor=3)
len(out)

# %%
results = []
for i in range(0, 36, 6):
    res = make_left_paddle_direction_prompt(out[i])
    results.append((i, res))

results
# %%
out[0].render()

# %%
