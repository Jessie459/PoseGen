import argparse
import glob
import os

import cv2
import imageio.v2 as imageio
import numpy as np
from tqdm import tqdm


def save_video(save_path: str, video: list[np.ndarray], quality=9, fps=30):
    if os.path.dirname(save_path):
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

    writer = imageio.get_writer(save_path, fps=fps, quality=quality)
    for frame in tqdm(video, desc="Saving video"):
        writer.append_data(frame)
    writer.close()


def load_video(video_path):
    import decord
    decord.bridge.set_bridge("native")

    video_reader = decord.VideoReader(video_path)
    video = video_reader[:].asnumpy()
    video = [frame for frame in video]
    del video_reader
    return video


def soft_append_frames(history: list[np.ndarray], current: list[np.ndarray], overlap: int = 0):
    if overlap == 0:
        return history + current

    weights = np.linspace(1, 0, num=overlap).tolist()
    blended_list = []
    _history = history[-overlap:]
    _current = current[:overlap]
    for i in range(overlap):
        blended = weights[i] * _history[i] + (1 - weights[i]) * _current[i]
        blended_list.append(blended.astype(np.uint8))
    output = history[:-overlap] + blended_list + current[overlap:]
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_path", type=str, required=True)
    parser.add_argument("--video_root", type=str, required=True)
    args = parser.parse_args()

    head_length = 21
    tail_length = 20

    cap = cv2.VideoCapture(args.video_path)
    orig_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Original length: {orig_length}")
    cap.release()

    video_root = args.video_root
    video_paths = glob.glob(os.path.join(video_root, "video-chunk-*.mp4"))
    video_paths.sort(key=lambda x: int(os.path.splitext(os.path.basename(x))[0].split("-")[-1]))
    print("Video chunks:")
    print("\n".join(video_paths))

    base_videos = []
    cond_videos = []

    for i in tqdm(range(len(video_paths))):
        if i % 2 == 0:
            base_videos.append(load_video(video_paths[i]))
        else:
            cond_videos.append(load_video(video_paths[i]))

    history = base_videos[0]
    base_videos = base_videos[1:]

    for i in tqdm(range(1, len(video_paths))):
        if i % 2 == 1:
            overlap = head_length
            current = cond_videos[0]
            cond_videos = cond_videos[1:]
        else:
            overlap = tail_length
            current = base_videos[0]
            base_videos = base_videos[1:]
        history = soft_append_frames(history, current, overlap=overlap)

    video = history[:orig_length]
    save_video(os.path.join(video_root, "video.mp4"), video, fps=30)


if __name__ == "__main__":
    main()
