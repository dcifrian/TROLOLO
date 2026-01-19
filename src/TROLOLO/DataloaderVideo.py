# Copyright 2025 Diego Cifrian Bueno
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import numpy as np
from torchvision.utils import make_grid
from torchvision.transforms.functional import resize as tv_resize
import imageio
from pathlib import Path

from dali_hard_mining import DALIHardMiningWrapper


def dataloader_to_video(
        dataloader,
        output_path,
        num_batches=None,
        fps=10,
        output_height=1080,
        nrow=None,
):
    """
    Create a video from dataloader batches showing image grids.

    Args:
        dataloader: PyTorch DataLoader yielding batches of images
        output_path: Path to save the output video
        num_batches: Number of batches to process (None = exhaust dataloader)
        fps: Frames per second for output video
        output_height: Target height in pixels (width computed to maintain aspect ratio)
        nrow: Number of images per row in grid (None = auto based on batch size)
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    frames = []

    for i, batch in enumerate(dataloader):
        if num_batches is not None and i >= num_batches:
            break

        # Handle both (images, labels) and just images
        images = batch[0] if isinstance(batch, (tuple, list)) else batch

        # Make grid (returns CHW tensor in [0, 1])
        if nrow is None:
            nrow = int(np.ceil(np.sqrt(images.shape[0])))

        grid = make_grid(images, nrow=nrow, normalize=True, padding=2)

        # Resize to target height (torchvision handles aspect ratio)
        h, w = grid.shape[1:]
        new_w = int(w * output_height / h)
        grid_resized = tv_resize(grid, [output_height, new_w], antialias=True)

        # Convert to HWC numpy array in [0, 255]
        grid_np = (grid_resized.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)

        frames.append(grid_np)

    # Write video
    import cv2

    # Replace the imageio section with:
    h, w = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (w, h))

    for frame in frames:
        # OpenCV uses BGR, numpy arrays from our code are RGB
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

    writer.release()
    print(f"Video saved to {output_path} ({len(frames)} frames at {fps} fps)")

if __name__ == "__main__":
    batch_size=16
    train_loader = DALIHardMiningWrapper(
        data_path="/home/pickman/Escritorio/TROLOLO/data/Imagenet/train",
        image_size=256,
        batch_size=batch_size,
        workers=16,
        num_classes=1000,
        one_hot=False
    )
    train_loader_len = 1280000 // batch_size
    dataloader_to_video(train_loader, "augmentations.mp4", num_batches=500)