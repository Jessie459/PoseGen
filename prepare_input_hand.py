import argparse
import os

import cv2
import numpy as np

from src.utils import save_video_cv2

import decord  # isort:skip


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--normal_path", type=str, required=True)
    parser.add_argument("--seg_path", type=str, required=True)
    parser.add_argument("--video_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()

    segs = np.load(args.seg_path)  # (f h w)
    normals = np.load(args.normal_path)  # (f h w c)

    results = []
    for seg, normal in zip(segs, normals):
        result = np.zeros_like(normal)
        mask = (seg == 5) | (seg == 14)
        result[mask, :] = normal[mask, :]
        results.append(result)

    vr = decord.VideoReader(args.video_path)
    video = vr[:].asnumpy()
    del vr

    viz_results = [(r * 0.5 + v * 0.5).astype(np.uint8) for r, v in zip(results, video)]
    viz_results = [cv2.cvtColor(x, cv2.COLOR_RGB2BGR) for x in viz_results]
    results = [cv2.cvtColor(x, cv2.COLOR_RGB2BGR) for x in results]

    save_video_cv2(os.path.join(args.output_dir, f"hand.mp4"), results, fps=30)
    save_video_cv2(os.path.join(args.output_dir, f"hand_viz.mp4"), viz_results, fps=30)


if __name__ == "__main__":
    main()
