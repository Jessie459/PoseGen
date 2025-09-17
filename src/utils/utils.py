import argparse
import os
import random
from typing import List, Union

import cv2
import imageio
import numpy as np
import torch
from tqdm import tqdm


__all__ = ["str2bool", "save_video", "save_video_cv2", "set_seed"]


def str2bool(v):
    """
    Convert a string to a boolean.

    Supported true values: 'yes', 'true', 't', 'y', '1'
    Supported false values: 'no', 'false', 'f', 'n', '0'

    Args:
        v (str): String to convert.

    Returns:
        bool: Converted boolean value.

    Raises:
        argparse.ArgumentTypeError: If the value cannot be converted to boolean.
    """
    if isinstance(v, bool):
        return v
    v_lower = v.lower()
    if v_lower in ("yes", "true", "t", "y", "1"):
        return True
    elif v_lower in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected (True/False)")


def save_video(save_path: str, video: Union[np.ndarray, List[np.ndarray]], quality=9, fps=30, backend="imageio"):
    """
    Args:
        save_path (str): Video save path.
        video (Union[np.ndarray, List[np.ndarray]]): Video to be saved.
        quality (int, optional): Video output quality. Uses variable bit rate. Highest quality is 10, lowest is 0.
        fps (int, optional): Frames per second.
        backend (str, optional): "imageio" or "cv2".
    """
    if os.path.dirname(save_path):
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

    if backend == "imageio":
        writer = imageio.get_writer(save_path, fps=fps, quality=quality)
        for frame in video:
            writer.append_data(frame)
        writer.close()

    elif backend == "cv2":
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        height, width, _ = video[0].shape
        video_writer = cv2.VideoWriter(save_path, fourcc, fps, (width, height))
        for frame in video:
            video_writer.write(frame)
        video_writer.release()

    else:
        raise ValueError(f"invalid backend: {backend}")


def save_video_cv2(save_path: str, frames: List[np.ndarray], fps=30):
    """
    Args:
        save_path (str): Video save path.
        frames (List[np.ndarray]): Video to be saved.
        fps (int, optional): Frames per second.
    """
    if os.path.dirname(save_path):
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

    H, W, _ = frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(save_path, fourcc, fps, (W, H))
    for frame in tqdm(frames, desc="Saving video"):
        out.write(frame)
    out.release()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
