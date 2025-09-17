import argparse
import os
import pickle

import cv2
import numpy as np
from src.utils import save_video_cv2
from src.visualization.visualizer import CocoWholebodyPoseVisualizer
from tqdm import tqdm


KPT_RANGE_L_HAND = (91, 112)  # 21 left hand keypoints (133-keypoint version)
KPT_RANGE_R_HAND = (112, 133)  # 21 right hand keypoints (133-keypoint version)
HEIGHT, WIDTH = 1280, 720


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pose_path", type=str, required=True)
    parser.add_argument("--video_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--body_kpt_thr", type=float, default=0.3)
    parser.add_argument("--hand_kpt_thr", type=float, default=0.7)
    args = parser.parse_args()

    hand_kpt_thr = args.hand_kpt_thr
    body_kpt_thr = args.body_kpt_thr
    visualizer = CocoWholebodyPoseVisualizer(hand_kpt_thr=hand_kpt_thr, body_kpt_thr=body_kpt_thr)

    cap = cv2.VideoCapture(args.video_path)
    assert cap.isOpened(), "Cannot open the video"

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    with open(args.pose_path, "rb") as f:
        kpt_list = pickle.load(f)
    assert len(kpt_list) == frame_count

    kpt_img_all = []
    viz_img_all = []

    for frame_idx in tqdm(range(frame_count)):
        kpt_per_frame = kpt_list[frame_idx]

        ret, frame = cap.read()
        assert ret, "Cannot read the frame"

        keypoints = [kpt_per_frame[0]["keypoints"]]
        keypoint_scores = [kpt_per_frame[0]["keypoint_scores"]]

        keypoints = np.stack(keypoints, axis=0)  # (1, 133, 2)
        keypoint_scores = np.stack(keypoint_scores, axis=0)  # (1, 133)

        kpt_img = visualizer(keypoints, keypoint_scores, image_size=[frame_height, frame_width])
        kpt_img_all.append(kpt_img)

        viz_img = (kpt_img * 0.6 + frame * 0.4).astype(np.uint8)
        viz_img_all.append(viz_img)

    cap.release()

    os.makedirs(args.output_dir, exist_ok=True)

    save_path = os.path.join(args.output_dir, "pose.mp4")
    save_video_cv2(save_path, kpt_img_all, fps=fps)

    save_path = os.path.join(args.output_dir, f"pose_viz.mp4")
    save_video_cv2(save_path, viz_img_all, fps=fps)


if __name__ == "__main__":
    main()
