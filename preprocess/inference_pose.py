import argparse
import os
import pickle
import warnings
from typing import List, Union

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from classes_and_palettes import (
    COCO_WHOLEBODY_KPTS_COLORS,
    COCO_WHOLEBODY_SKELETON_INFO,
)
from inference_utils import save_video
from pose_utils import top_down_affine_transform, udp_decode
from tqdm import tqdm

try:
    from mmdet.apis import inference_detector, init_detector
    from mmdet.structures import DetDataSample, SampleList
    from mmdet.utils import get_test_pipeline_cfg
    has_mmdet = True
except (ImportError, ModuleNotFoundError):
    has_mmdet = False


warnings.filterwarnings("ignore")


KPTS_COLORS = COCO_WHOLEBODY_KPTS_COLORS  ## 133 keypoints
SKELETON_INFO = COCO_WHOLEBODY_SKELETON_INFO


def preprocess_pose(orig_img, bboxes_list, input_shape, mean, std):
    """Preprocess pose images and bboxes."""
    preprocessed_images = []
    centers = []
    scales = []
    for bbox in bboxes_list:
        img, center, scale = top_down_affine_transform(orig_img.copy(), bbox)
        img = cv2.resize(img, (input_shape[1], input_shape[0]), interpolation=cv2.INTER_LINEAR).transpose(2, 0, 1)
        img = torch.from_numpy(img)
        img = img[[2, 1, 0], ...].float()
        mean = torch.Tensor(mean).view(-1, 1, 1)
        std = torch.Tensor(std).view(-1, 1, 1)
        img = (img - mean) / std
        preprocessed_images.append(img)
        centers.extend(center)
        scales.extend(scale)
    return preprocessed_images, centers, scales


def batch_inference_topdown(model: nn.Module, imgs: List[Union[np.ndarray, str]], dtype=torch.bfloat16, flip=False):
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=dtype):
        heatmaps = model(imgs.cuda())
        if flip:
            heatmaps_ = model(imgs.to(dtype).cuda().flip(-1))
            heatmaps = (heatmaps + heatmaps_) * 0.5
        imgs.cpu()
    return heatmaps.cpu()


def process_results(img, results, input_shape, heatmap_scale, kpt_colors, kpt_thr, radius, skeleton_info, thickness):
    heatmap = results["heatmaps"]
    centres = results["centres"]
    scales = results["scales"]

    instance_keypoints = []
    instance_scores = []

    for i in range(len(heatmap)):
        result = udp_decode(
            heatmap[i].cpu().unsqueeze(0).float().data[0].numpy(),
            input_shape,
            (int(input_shape[0] / heatmap_scale), int(input_shape[1] / heatmap_scale)),
        )

        keypoints, keypoint_scores = result
        keypoints = (keypoints / input_shape) * scales[i] + centres[i] - 0.5 * scales[i]
        instance_keypoints.append(keypoints[0])
        instance_scores.append(keypoint_scores[0])

    res = [
        {"keypoints": keypoints.tolist(), "keypoint_scores": keypoint_scores.tolist()}
        for keypoints, keypoint_scores in zip(instance_keypoints, instance_scores)
    ]

    instance_keypoints = np.array(instance_keypoints).astype(np.float32)
    instance_scores = np.array(instance_scores).astype(np.float32)

    keypoints_visible = np.ones(instance_keypoints.shape[:-1])
    for kpts, score, visible in zip(instance_keypoints, instance_scores, keypoints_visible):
        kpts = np.array(kpts, copy=False)

        if kpt_colors is None or isinstance(kpt_colors, str) or len(kpt_colors) != len(kpts):
            raise ValueError(
                f"the length of kpt_color " f"({len(kpt_colors)}) does not matches " f"that of keypoints ({len(kpts)})"
            )

        # draw each point on image
        for kid, kpt in enumerate(kpts):
            if score[kid] < kpt_thr or not visible[kid] or kpt_colors[kid] is None:
                # skip the point that should not be drawn
                continue

            color = kpt_colors[kid]
            if not isinstance(color, str):
                color = tuple(int(c) for c in color[::-1])
            img = cv2.circle(img, (int(kpt[0]), int(kpt[1])), int(radius), color, -1)

        # draw skeleton
        for skid, link_info in skeleton_info.items():
            pt1_idx, pt2_idx = link_info["link"]
            color = link_info["color"][::-1]  # BGR

            pt1 = kpts[pt1_idx]
            pt1_score = score[pt1_idx]
            pt2 = kpts[pt2_idx]
            pt2_score = score[pt2_idx]

            if pt1_score > kpt_thr and pt2_score > kpt_thr:
                x1_coord = int(pt1[0])
                y1_coord = int(pt1[1])
                x2_coord = int(pt2[0])
                y2_coord = int(pt2[1])
                cv2.line(img, (x1_coord, y1_coord), (x2_coord, y2_coord), color, thickness=thickness)

    return res, img


def fake_pad_images_to_batchsize(imgs, batch_size):
    return F.pad(imgs, (0, 0, 0, 0, 0, 0, 0, batch_size - imgs.shape[0]), value=0)


def load_model(checkpoint, use_torchscript=False):
    if use_torchscript:
        return torch.jit.load(checkpoint)
    else:
        return torch.export.load(checkpoint).module()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=8, help="Set batch size to do batch inference. ")
    parser.add_argument("--shape", type=int, nargs="+", default=[1024, 768], help="input image size (height, width)")
    parser.add_argument("--fp16", action="store_true", default=False, help="Model inference dtype")
    parser.add_argument("--det-cat-id", type=int, default=0, help="Category id for bounding box detection model")
    parser.add_argument("--bbox-thr", type=float, default=0.3, help="Bounding box score threshold")
    parser.add_argument("--nms-thr", type=float, default=0.3, help="IoU threshold for bounding box NMS")
    parser.add_argument("--kpt-thr", type=float, default=0.3, help="Visualizing keypoint thresholds")
    parser.add_argument("--radius", type=int, default=2, help="Keypoint radius for visualization")
    parser.add_argument("--thickness", type=int, default=2, help="Keypoint skeleton thickness for visualization")
    parser.add_argument("--heatmap-scale", type=int, default=4, help="Heatmap scale for keypoints. Image to heatmap ratio")
    parser.add_argument("--flip", type=bool, default=False, help="Flip the input image horizontally and inference again")
    args = parser.parse_args()

    args.video_path = os.path.abspath(os.path.expanduser(args.video_path))
    args.output_dir = os.path.abspath(os.path.expanduser(args.output_dir))

    use_det = True
    assert has_mmdet, "Please install mmdet to run the demo."

    from detector_utils import (
        adapt_mmdet_pipeline,
        init_detector,
        process_images_detector,
    )

    assert len(args.shape) == 2
    input_shape = (3,) + tuple(args.shape)  # (3, 1024, 768)

    torch._inductor.config.force_fuse_int_mm_with_mul = True
    torch._inductor.config.use_mixed_mm = True

    # Build detector
    if use_det:
        det_config = "../pose/demo/mmdetection_cfg/rtmdet_m_640-8xb32_coco-person_no_nms.py"
        det_checkpoint = os.path.join(
            os.environ["SAPIENS_CKPT_ROOT"],
            "detector",
            "rtmdet_m_8xb32-100e_coco-obj365-person-235e8209.pth",
        )
        detector = init_detector(det_config, det_checkpoint, device="cuda")
        detector.cfg = adapt_mmdet_pipeline(detector.cfg)

    # Build pose estimator
    pose_checkpoint = os.path.join(
        os.environ["SAPIENS_CKPT_ROOT"],
        "pose",
        "sapiens_2b_coco_wholebody_best_coco_wholebody_AP_745_torchscript.pt2",
    )
    USE_TORCHSCRIPT = "_torchscript" in pose_checkpoint
    pose_estimator = load_model(pose_checkpoint, USE_TORCHSCRIPT)
    if not USE_TORCHSCRIPT:
        dtype = torch.half if args.fp16 else torch.bfloat16
        pose_estimator.to(dtype)
        pose_estimator = torch.compile(pose_estimator, mode="max-autotune", fullgraph=True)
    else:
        dtype = torch.float32  # TorchScript models use float32
        pose_estimator = pose_estimator.to("cuda")

    cap = cv2.VideoCapture(args.video_path)
    assert cap.isOpened(), "Cannot open the video"
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    res_all = []
    viz_all = []

    batch_size = args.batch_size

    for batch_start_idx in tqdm(range(0, num_frames, batch_size)):
        batch_orig_imgs = []
        for _ in range(batch_size):
            ret, frame = cap.read()
            if not ret:
                break
            batch_orig_imgs.append(frame)
        if not batch_orig_imgs:
            break
        batch_orig_imgs = np.stack(batch_orig_imgs)  # [b h w c]

        orig_img_shape = batch_orig_imgs.shape
        valid_images_len = len(batch_orig_imgs)

        if use_det:
            imgs = batch_orig_imgs.copy()[..., [2, 1, 0]]  # RGB
            bboxes_batch = process_images_detector(args, imgs, detector)
        else:
            bboxes_batch = [[] for _ in range(len(batch_orig_imgs))]

        assert len(bboxes_batch) == valid_images_len

        for i, bboxes in enumerate(bboxes_batch):  # (x0, y0, x1, y1)
            if len(bboxes) == 0:
                bboxes_batch[i] = np.array([[0, 0, orig_img_shape[2], orig_img_shape[1]]])

        img_bbox_map = {}
        for i, bboxes in enumerate(bboxes_batch):
            img_bbox_map[i] = len(bboxes)

        pose_ops = []
        for i, bbox_list in zip(batch_orig_imgs, bboxes_batch):
            pose_ops.append(
                preprocess_pose(
                    i, bbox_list, (input_shape[1], input_shape[2]), [123.5, 116.5, 103.5], [58.5, 57.0, 57.5]
                )
            )

        pose_imgs, pose_img_centers, pose_img_scales = [], [], []
        for op in pose_ops:
            pose_imgs.extend(op[0])
            pose_img_centers.extend(op[1])
            pose_img_scales.extend(op[2])

        n_pose_batches = (len(pose_imgs) + batch_size - 1) // batch_size

        torch.compiler.cudagraph_mark_step_begin()

        pose_results = []
        for i in range(n_pose_batches):
            imgs = torch.stack(pose_imgs[i * batch_size : (i + 1) * batch_size], dim=0)
            valid_len = len(imgs)
            imgs = fake_pad_images_to_batchsize(imgs, batch_size)
            pose_results.extend(batch_inference_topdown(pose_estimator, imgs, dtype=dtype)[:valid_len])

        batched_results = []
        for _, bbox_len in img_bbox_map.items():
            result = {
                "heatmaps": pose_results[:bbox_len].copy(),
                "centres": pose_img_centers[:bbox_len].copy(),
                "scales": pose_img_scales[:bbox_len].copy(),
            }
            batched_results.append(result)
            del (pose_results[:bbox_len], pose_img_centers[:bbox_len], pose_img_scales[:bbox_len])

        assert len(batched_results) == len(batch_orig_imgs)

        for i, r in zip(batch_orig_imgs[:valid_images_len], batched_results[:valid_images_len]):
            res, viz = process_results(
                i.copy(),
                r,
                (input_shape[2], input_shape[1]),
                args.heatmap_scale,
                KPTS_COLORS,
                args.kpt_thr,
                args.radius,
                SKELETON_INFO,
                args.thickness,
            )
            res_all.append(res)
            viz_all.append(viz)

    cap.release()

    os.makedirs(args.output_dir, exist_ok=True)

    save_path = os.path.join(args.output_dir, "pose.pkl")
    with open(save_path, "wb") as f:
        pickle.dump(res_all, f)

    save_path = os.path.join(args.output_dir, "pose.mp4")
    save_video(save_path, viz_all, fps=fps)


if __name__ == "__main__":
    main()
