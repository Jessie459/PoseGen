import os

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm


class InferenceDataset(Dataset):
    def __init__(self, frames, shape=(1024, 768), mean=(123.5, 116.5, 103.5), std=(58.5, 57.0, 57.5)):
        self.frames = frames
        self.shape = shape
        self.mean = torch.tensor(mean) if mean is not None else None
        self.std = torch.tensor(std) if std is not None else None

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, index):
        frame = self.frames[index]  # RGB
        if self.shape is not None:
            frame = cv2.resize(frame, (self.shape[1], self.shape[0]), interpolation=cv2.INTER_LINEAR)
        frame = torch.from_numpy(frame).permute(2, 0, 1).float()
        if self.mean is not None and self.std is not None:
            mean = self.mean.view(-1, 1, 1)
            std = self.std.view(-1, 1, 1)
            frame = (frame - mean) / std
        return frame


def save_video(save_path: str, frames: list[np.ndarray], fps=30):
    H, W, _ = frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(save_path, fourcc, fps, (W, H))
    for frame in tqdm(frames, desc="Saving video"):
        out.write(frame)
    out.release()
