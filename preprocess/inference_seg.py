import argparse
import os

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from classes_and_palettes import GOLIATH_CLASSES, GOLIATH_PALETTE
from inference_utils import InferenceDataset, save_video
from torch.utils.data import DataLoader
from tqdm import tqdm

import decord  # isort:skip


torchvision.disable_beta_transforms_warning()


def inference_model(model, imgs, dtype=torch.bfloat16):
    with torch.no_grad():
        results = model(imgs.to(dtype).cuda())
        imgs.cpu()
    results = [r.cpu() for r in results]
    return results


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
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--fp16", action="store_true", default=False)
    args = parser.parse_args()

    args.video_path = os.path.abspath(os.path.expanduser(args.video_path))
    args.output_dir = os.path.abspath(os.path.expanduser(args.output_dir))

    torch._inductor.config.force_fuse_int_mm_with_mul = True
    torch._inductor.config.use_mixed_mm = True

    CHECKPOINT = os.path.join(
        os.environ["SAPIENS_CKPT_ROOT"],
        "seg",
        "sapiens_1b_goliath_best_goliath_mIoU_7994_epoch_151_torchscript.pt2",
    )
    USE_TORCHSCRIPT = "_torchscript" in CHECKPOINT
    assert USE_TORCHSCRIPT
    exp_model = load_model(CHECKPOINT, USE_TORCHSCRIPT)

    if not USE_TORCHSCRIPT:
        dtype = torch.half if args.fp16 else torch.bfloat16
        exp_model.to(dtype)
        exp_model = torch.compile(exp_model, mode="max-autotune", fullgraph=True)
    else:
        dtype = torch.float32  # TorchScript models use float32
        exp_model = exp_model.cuda()

    video_reader = decord.VideoReader(args.video_path)
    frames = video_reader[:].asnumpy()
    del video_reader

    dataset = InferenceDataset(frames)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False)

    classes = GOLIATH_CLASSES
    palette = GOLIATH_PALETTE
    opacity = 0.5
    threshold = 0.3

    seg_all = []
    viz_all = []

    for batch_idx, batch_imgs in enumerate(tqdm(dataloader)):
        batch_imgs_len = len(batch_imgs)
        batch_imgs = fake_pad_images_to_batchsize(batch_imgs, args.batch_size)
        batch_result = inference_model(exp_model, batch_imgs, dtype=dtype)

        batch_frames = frames[batch_idx * args.batch_size : (batch_idx + 1) * args.batch_size]
        for img, result in zip(batch_frames[:batch_imgs_len], batch_result[:batch_imgs_len]):
            seg_logits = F.interpolate(result.unsqueeze(0), size=img.shape[:2], mode="bilinear").squeeze(0)
            if seg_logits.shape[0] > 1:
                pred_sem_seg = seg_logits.argmax(dim=0, keepdim=True)
            else:
                seg_logits = seg_logits.sigmoid()
                pred_sem_seg = (seg_logits > threshold).to(seg_logits)
            pred_sem_seg = pred_sem_seg.data[0].numpy()
            seg_all.append(pred_sem_seg)

            sem_seg = pred_sem_seg
            ids = np.unique(sem_seg)[::-1]
            legal_indices = ids < len(classes)
            ids = ids[legal_indices]
            labels = np.array(ids, dtype=np.int64)

            colors = [palette[label] for label in labels]

            mask = np.zeros_like(img)
            for label, color in zip(labels, colors):
                mask[sem_seg == label, :] = color
            viz = (img * (1 - opacity) + mask * opacity).astype(np.uint8)
            viz = cv2.cvtColor(viz, cv2.COLOR_RGB2BGR)
            viz_all.append(viz)

    seg_all = np.stack(seg_all)
    viz_all = np.stack(viz_all)

    os.makedirs(args.output_dir, exist_ok=True)

    save_path = os.path.join(args.output_dir, "seg.npy")
    np.save(save_path, seg_all)

    save_path = os.path.join(args.output_dir, "seg.mp4")
    save_video(save_path, viz_all, fps=30)


if __name__ == "__main__":
    main()
