import argparse
import json
import math
import os

import cv2
import imageio
import numpy as np
import torch
import torchvision.transforms as T
from einops import rearrange
from src.models import ModelManager
from src.pipelines.wan_video_kvshare import WanVideoPipeline
from src.utils import save_video, str2bool

import decord  # isort:skip
decord.bridge.set_bridge("torch")

os.environ["TOKENIZERS_PARALLELISM"] = "false"

HEIGHT, WIDTH = 1280, 720
TRANSFORM = T.Compose(
    [
        T.RandomResizedCrop(size=(HEIGHT, WIDTH), scale=(1.0, 1.0), ratio=(WIDTH / HEIGHT, WIDTH / HEIGHT)),
        T.Lambda(lambda x: (x / 255.0) * 2.0 - 1.0),
    ]
)
TOKEN_DICT = {"woman": 27502, "girl": 15146, "man": 621, "boy": 18942}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, choices=["anch", "base", "cond"], required=True)
    parser.add_argument("--pose_path", type=str, required=True)
    parser.add_argument("--hand_path", type=str, required=True)
    parser.add_argument("--image_path", type=str, required=True)
    parser.add_argument("--prompt_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--use_hand", type=str2bool, nargs="?", const=True, default=True)
    parser.add_argument("--pad", type=str2bool, default=True)
    parser.add_argument("--num_chunks", type=int, default=-1)
    parser.add_argument("--head_length", type=int, default=21)
    parser.add_argument("--tail_length", type=int, default=20)
    parser.add_argument("--num_frames", type=int, default=81)
    parser.add_argument("--num_inference_steps", type=int, default=25)
    parser.add_argument("--cfg_scale", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fps", type=float, default=30)
    parser.add_argument("--use_attn_mask", type=str2bool, nargs="?", const=True, default=True)
    parser.add_argument("--max_block_idx", type=int, default=10)
    parser.add_argument("--anch_chunk_idx", type=int, default=0)
    parser.add_argument("-s", "--kvshare_step_range", type=int, nargs="+", default=[0, 2])
    parser.add_argument("-b", "--kvshare_block_range", type=int, nargs="+", default=[34, 40])
    parser.add_argument("-p", "--num_persistent_param_in_dit", type=str, default=None)
    return parser.parse_args()


def prepare_image(image_path):
    image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
    image = torch.from_numpy(image).unsqueeze(0)
    image = rearrange(image, "f h w c -> f c h w").contiguous()
    image = TRANSFORM(image)
    image = rearrange(image, "f c h w -> c f h w").contiguous()
    return image


def prepare_cond(pose_path, hand_path, num_frames, num_chunks, head_length, tail_length, pad):
    pose_reader = decord.VideoReader(pose_path)
    orig_length = len(pose_reader)
    print(f"Original pose video length: {orig_length}")

    if num_chunks <= 0:
        fn = math.ceil if pad else math.floor
        num_chunks = int(fn((orig_length - num_frames) / (2 * num_frames - head_length - tail_length)))
        num_chunks = num_chunks * 2 + 1
        print(f"`num_chunks` is automatically set to {num_chunks}")

    if num_chunks % 2 == 0:
        num_chunks -= 1
        print(f"`num_chunks` should be an odd number, adjusted to {num_chunks}")

    if num_chunks <= 0:
        raise RuntimeError(f"No video chunks to generate")

    pose_length = num_chunks * num_frames - num_chunks // 2 * (head_length + tail_length)
    print(f"Selected pose video length: {pose_length}")

    if hand_path is not None:
        hand_reader = decord.VideoReader(hand_path)
    else:
        hand_reader = None

    return num_chunks, orig_length, pose_reader, hand_reader


def load_video(video_path):
    video_reader = decord.VideoReader(video_path)
    video = video_reader[:].numpy()
    video = [frame for frame in video]
    del video_reader
    return video


def load_dit(ckpt_path, device, torch_dtype, kvshare_kwargs=None, lora_alpha=1.0):
    from src.models.utils import init_weights_on_device, merge_lora_weights
    from src.models.wan_video_dit_kvshare import WanModel

    model_config_path = os.path.join(ckpt_path, "model_config.json")
    with open(model_config_path, "r") as f:
        model_config = json.load(f)

    model_config["kvshare_kwargs"] = kvshare_kwargs

    with init_weights_on_device():
        model = WanModel(**model_config)

    state_dict = merge_lora_weights(ckpt_path=ckpt_path, lora_alpha=lora_alpha)
    model.load_state_dict(state_dict, assign=True)

    model = model.to(dtype=torch_dtype, device=device)
    model.eval()
    return model


def main():
    args = parse_args()

    num_frames = args.num_frames
    num_chunks = args.num_chunks
    head_length = args.head_length
    tail_length = args.tail_length
    assert head_length == 21 and tail_length == 20

    if args.mode == "cond":
        # Turn off kv-sharing
        args.use_attn_mask = False
        args.token_indices = None
        args.max_block_idx = None
        args.kvshare_step_range = None
        args.kvshare_block_range = None

    torch_dtype = torch.bfloat16

    vae_path = os.path.join(os.environ["POSEGEN_CKPT_PATH"], "Wan2.1_VAE.pth")
    text_encoder_path = os.path.join(os.environ["POSEGEN_CKPT_PATH"], "models_t5_umt5-xxl-enc-bf16.pth")
    image_encoder_path = os.path.join(os.environ["POSEGEN_CKPT_PATH"], "models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth")

    model_manager = ModelManager(device="cpu", torch_dtype=torch_dtype)
    model_manager.load_models([vae_path, text_encoder_path, image_encoder_path])
    pipe = WanVideoPipeline.from_model_manager(model_manager, device="cuda")

    if args.mode != "cond":
        kvshare_kwargs = {
            "steps": list(range(*args.kvshare_step_range)) if args.kvshare_step_range else [],
            "blocks": list(range(*args.kvshare_block_range)) if args.kvshare_block_range else [],
        }
        ckpt_path = os.path.join(os.environ["POSEGEN_CKPT_PATH"], "720p-81f-base")
        pipe.dit = load_dit(ckpt_path, device="cpu", torch_dtype=torch_dtype, kvshare_kwargs=kvshare_kwargs)
    else:
        ckpt_path = os.path.join(os.environ["POSEGEN_CKPT_PATH"], "720p-81f-cond")
        pipe.dit = load_dit(ckpt_path, device="cpu", torch_dtype=torch_dtype)

    num_persistent_param_in_dit = args.num_persistent_param_in_dit
    if num_persistent_param_in_dit is not None:
        num_persistent_param_in_dit = int(eval(num_persistent_param_in_dit))
    pipe.enable_vram_management(num_persistent_param_in_dit=num_persistent_param_in_dit)

    with open(args.prompt_path, "r") as f:
        prompt_list = f.readlines()
    if len(prompt_list) == 1:
        prompt = prompt_list[0]
    else:
        prompt = max(prompt_list, key=len)
    print(f"Prompt:\n{prompt}")

    if args.use_attn_mask:
        with open(args.prompt_path.replace(".txt", ".json"), "r") as f:
            prompt_token_indices = json.load(f)
        for keyword in TOKEN_DICT.keys():
            if keyword in prompt:
                args.token_indices = [prompt_token_indices.index(TOKEN_DICT[keyword])]
                break
        assert args.token_indices is not None
        print(f"[{args.mode}] attention mask is enabled")
    else:
        args.token_indices = None
        print(f"[{args.mode}] attention mask is disabled")

    input_image = prepare_image(image_path=args.image_path)

    num_chunks, orig_length, pose_reader, hand_reader = prepare_cond(
        pose_path=args.pose_path,
        hand_path=args.hand_path if args.use_hand else None,
        num_frames=num_frames,
        num_chunks=num_chunks,
        head_length=head_length,
        tail_length=tail_length,
        pad=args.pad,
    )
    print("Number of chunks:", num_chunks)

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    with open(os.path.join(output_dir, f"config_{args.mode}.json"), "w") as f:
        json.dump(vars(args), f, indent=4)

    # viz_image = rearrange(input_image, "c f h w -> f h w c").contiguous()[0]
    # viz_image = (viz_image * 0.5 + 0.5) * 255
    # viz_image = viz_image.cpu().numpy().astype(np.uint8)
    # imageio.imsave(os.path.join(output_dir, "image.png"), viz_image)

    # viz_pose = rearrange(pose_video, "c f h w -> f h w c").contiguous()
    # viz_pose = (viz_pose * 0.5 + 0.5) * 255
    # viz_pose = viz_pose.cpu().numpy().astype(np.uint8)
    # save_video(os.path.join(output_dir, "video_pose.mp4"), viz_pose, fps=args.fps)

    # if args.use_hand:
    #     viz_hand = rearrange(hand_video, "c f h w -> f h w c").contiguous()
    #     viz_hand = (viz_hand * 0.5 + 0.5) * 255
    #     viz_hand = viz_hand.cpu().numpy().astype(np.uint8)
    #     save_video(os.path.join(output_dir, "video_hand.mp4"), viz_hand, fps=args.fps)

    negative_prompt = "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"

    anch_chunk_indices = [args.anch_chunk_idx]
    base_chunk_indices = list(range(0, num_chunks, 2))
    cond_chunk_indices = list(range(1, num_chunks, 2))
    print(f"Chunk indices (anch): {anch_chunk_indices}")
    print(f"Chunk indices (base): {base_chunk_indices}")
    print(f"Chunk indices (cond): {cond_chunk_indices}")

    if args.mode == "anch" or args.mode == "base":
        chunk_indices = anch_chunk_indices if args.mode == "anch" else base_chunk_indices
        for chunk_idx in chunk_indices:
            if args.mode == "base" and chunk_idx == args.anch_chunk_idx:
                continue
            print(f"[{args.mode}] current chunk: {chunk_idx}")

            start_idx = chunk_idx * num_frames - math.ceil(chunk_idx / 2) * head_length - math.floor(chunk_idx / 2) * tail_length
            end_idx = start_idx + num_frames

            if end_idx >= orig_length:
                # the start index of the previous chunk is always within bound, but not the current one
                pre_chunk_idx = chunk_idx - 1
                pre_start_idx = (
                    pre_chunk_idx * num_frames
                    - math.ceil(pre_chunk_idx / 2) * head_length
                    - math.floor(pre_chunk_idx / 2) * tail_length
                )

                pose_video = pose_reader.get_batch(range(pre_start_idx, orig_length))
                pad_length = end_idx - orig_length
                pose_video = torch.cat([pose_video, torch.flip(pose_video[-pad_length:], dims=(0,))], dim=0)[-num_frames:]
                pose_video = rearrange(pose_video, "f h w c -> f c h w").contiguous()
                pose_video = TRANSFORM(pose_video)
                pose_video = rearrange(pose_video, "f c h w -> c f h w").contiguous()

                hand_video = hand_reader.get_batch(range(pre_start_idx, orig_length))
                pad_length = end_idx - orig_length
                hand_video = torch.cat([hand_video, torch.flip(hand_video[-pad_length:], dims=(0,))], dim=0)[-num_frames:]
                hand_video = rearrange(hand_video, "f h w c -> f c h w").contiguous()
                hand_video = TRANSFORM(hand_video)
                hand_video = rearrange(hand_video, "f c h w -> c f h w").contiguous()
            else:
                pose_video = pose_reader.get_batch(range(start_idx, end_idx))
                assert len(pose_video) == num_frames, f"pose video length {len(pose_video)} != 81"
                pose_video = rearrange(pose_video, "f h w c -> f c h w").contiguous()
                pose_video = TRANSFORM(pose_video)
                pose_video = rearrange(pose_video, "f c h w -> c f h w").contiguous()

                hand_video = hand_reader.get_batch(range(start_idx, end_idx))
                assert len(hand_video) == num_frames, f"hand video length {len(hand_video)} != 81"
                hand_video = rearrange(hand_video, "f h w c -> f c h w").contiguous()
                hand_video = TRANSFORM(hand_video)
                hand_video = rearrange(hand_video, "f c h w -> c f h w").contiguous()

            video = pipe(
                input_image=input_image,
                input_pose_video=pose_video,
                input_hand_video=hand_video,
                head_frames=None,
                tail_frames=None,
                prompt=prompt,
                negative_prompt=negative_prompt,
                height=HEIGHT,
                width=WIDTH,
                num_frames=num_frames,
                num_inference_steps=args.num_inference_steps,
                cfg_scale=args.cfg_scale,
                seed=args.seed,
                tiled=True,
                qkv_save_dir=os.path.join(output_dir, "qkv") if chunk_idx == args.anch_chunk_idx and args.mode != "cond" else None,
                qkv_load_dir=os.path.join(output_dir, "qkv") if chunk_idx != args.anch_chunk_idx and args.mode != "cond" else None,
                use_attn_mask=args.use_attn_mask,
                token_indices=args.token_indices if args.use_attn_mask else None,
                max_block_idx=args.max_block_idx if args.use_attn_mask else None,
                attn_maps_dir=os.path.join(output_dir, f"attn_maps_chunk{chunk_idx}") if args.use_attn_mask else None,
            )
            save_video(os.path.join(output_dir, f"video-chunk-{chunk_idx:04d}.mp4"), video, fps=args.fps)

    else:
        for chunk_idx in cond_chunk_indices:
            print(f"[{args.mode}] current chunk: {chunk_idx}")

            start_idx = chunk_idx * num_frames - math.ceil(chunk_idx / 2) * head_length - math.floor(chunk_idx / 2) * tail_length
            end_idx = min(start_idx + num_frames, orig_length)

            pose_video = pose_reader.get_batch(range(start_idx, end_idx))
            if len(pose_video) < num_frames:
                pad_length = num_frames - len(pose_video)
                pose_video = torch.cat([pose_video, torch.flip(pose_video[-pad_length:], dims=(0,))], dim=0)
            pose_video = rearrange(pose_video, "f h w c -> f c h w").contiguous()
            pose_video = TRANSFORM(pose_video)
            pose_video = rearrange(pose_video, "f c h w -> c f h w").contiguous()

            assert args.use_hand
            hand_video = hand_reader.get_batch(range(start_idx, end_idx))
            if len(hand_video) < num_frames:
                pad_length = num_frames - len(hand_video)
                hand_video = torch.cat([hand_video, torch.flip(hand_video[-pad_length:], dims=(0,))], dim=0)
            hand_video = rearrange(hand_video, "f h w c -> f c h w").contiguous()
            hand_video = TRANSFORM(hand_video)
            hand_video = rearrange(hand_video, "f c h w -> c f h w").contiguous()

            head_frames = load_video(os.path.join(output_dir, f"video-chunk-{(chunk_idx - 1):04d}.mp4"))[-head_length:]
            tail_frames = load_video(os.path.join(output_dir, f"video-chunk-{(chunk_idx + 1):04d}.mp4"))[:tail_length]

            video = pipe(
                input_image=input_image,
                input_pose_video=pose_video,
                input_hand_video=hand_video,
                head_frames=head_frames,
                tail_frames=tail_frames,
                prompt=prompt,
                negative_prompt=negative_prompt,
                height=HEIGHT,
                width=WIDTH,
                num_frames=num_frames,
                num_inference_steps=args.num_inference_steps,
                cfg_scale=args.cfg_scale,
                seed=args.seed,
                tiled=True,
            )
            save_video(os.path.join(output_dir, f"video-chunk-{chunk_idx:04d}.mp4"), video, fps=args.fps)

    print(f"[{args.mode}] Chunked videos are generated")


if __name__ == "__main__":
    main()
