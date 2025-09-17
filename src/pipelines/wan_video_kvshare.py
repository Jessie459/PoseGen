import os
from typing import Union

import numpy as np
import torch
from diffsynth.models.wan_video_image_encoder import WanImageEncoder
from diffsynth.models.wan_video_text_encoder import T5LayerNorm, T5RelativeEmbedding, WanTextEncoder
from diffsynth.models.wan_video_vae import CausalConv3d, RMS_norm, Upsample, WanVideoVAE
from diffsynth.prompters import WanPrompter
from diffsynth.schedulers.flow_match import FlowMatchScheduler
from diffsynth.vram_management import AutoWrappedLinear, AutoWrappedModule, enable_vram_management
from einops import rearrange
from PIL import Image
from tqdm import tqdm

from ..models.wan_video_dit_kvshare import RMSNorm, WanModel
from .base import BasePipeline


class WanVideoPipeline(BasePipeline):
    def __init__(self, device="cuda", torch_dtype=torch.float16, tokenizer_path=None):
        super().__init__(device=device, torch_dtype=torch_dtype)
        self.scheduler = FlowMatchScheduler(shift=5, sigma_min=0.0, extra_one_step=True)
        self.prompter = WanPrompter(tokenizer_path=tokenizer_path)
        self.text_encoder: WanTextEncoder = None
        self.image_encoder: WanImageEncoder = None
        self.dit: WanModel = None
        self.vae: WanVideoVAE = None
        self.model_names = ["text_encoder", "dit", "vae", "image_encoder"]
        self.height_division_factor = 16
        self.width_division_factor = 16
        self.use_unified_sequence_parallel = False

    def enable_vram_management(self, num_persistent_param_in_dit=None):
        dtype = next(iter(self.text_encoder.parameters())).dtype
        enable_vram_management(
            self.text_encoder,
            module_map={
                torch.nn.Linear: AutoWrappedLinear,
                torch.nn.Embedding: AutoWrappedModule,
                T5RelativeEmbedding: AutoWrappedModule,
                T5LayerNorm: AutoWrappedModule,
            },
            module_config=dict(
                offload_dtype=dtype,
                offload_device="cpu",
                onload_dtype=dtype,
                onload_device="cpu",
                computation_dtype=self.torch_dtype,
                computation_device=self.device,
            ),
        )
        dtype = next(iter(self.dit.parameters())).dtype
        enable_vram_management(
            self.dit,
            module_map={
                torch.nn.Linear: AutoWrappedLinear,
                torch.nn.Conv3d: AutoWrappedModule,
                torch.nn.LayerNorm: AutoWrappedModule,
                RMSNorm: AutoWrappedModule,
            },
            module_config=dict(
                offload_dtype=dtype,
                offload_device="cpu",
                onload_dtype=dtype,
                onload_device=self.device,
                computation_dtype=self.torch_dtype,
                computation_device=self.device,
            ),
            max_num_param=num_persistent_param_in_dit,
            overflow_module_config=dict(
                offload_dtype=dtype,
                offload_device="cpu",
                onload_dtype=dtype,
                onload_device="cpu",
                computation_dtype=self.torch_dtype,
                computation_device=self.device,
            ),
        )
        dtype = next(iter(self.vae.parameters())).dtype
        enable_vram_management(
            self.vae,
            module_map={
                torch.nn.Linear: AutoWrappedLinear,
                torch.nn.Conv2d: AutoWrappedModule,
                RMS_norm: AutoWrappedModule,
                CausalConv3d: AutoWrappedModule,
                Upsample: AutoWrappedModule,
                torch.nn.SiLU: AutoWrappedModule,
                torch.nn.Dropout: AutoWrappedModule,
            },
            module_config=dict(
                offload_dtype=dtype,
                offload_device="cpu",
                onload_dtype=dtype,
                onload_device=self.device,
                computation_dtype=self.torch_dtype,
                computation_device=self.device,
            ),
        )
        if self.image_encoder is not None:
            dtype = next(iter(self.image_encoder.parameters())).dtype
            enable_vram_management(
                self.image_encoder,
                module_map={
                    torch.nn.Linear: AutoWrappedLinear,
                    torch.nn.Conv2d: AutoWrappedModule,
                    torch.nn.LayerNorm: AutoWrappedModule,
                },
                module_config=dict(
                    offload_dtype=dtype,
                    offload_device="cpu",
                    onload_dtype=dtype,
                    onload_device="cpu",
                    computation_dtype=dtype,
                    computation_device=self.device,
                ),
            )
        self.enable_cpu_offload()

    def enable_dit_vram_management(self, num_persistent_param_in_dit=None):
        dtype = next(iter(self.dit.parameters())).dtype
        enable_vram_management(
            self.dit,
            module_map={
                torch.nn.Linear: AutoWrappedLinear,
                torch.nn.Conv3d: AutoWrappedModule,
                torch.nn.LayerNorm: AutoWrappedModule,
                RMSNorm: AutoWrappedModule,
            },
            module_config=dict(
                offload_dtype=dtype,
                offload_device="cpu",
                onload_dtype=dtype,
                onload_device=self.device,
                computation_dtype=self.torch_dtype,
                computation_device=self.device,
            ),
            max_num_param=num_persistent_param_in_dit,
            overflow_module_config=dict(
                offload_dtype=dtype,
                offload_device="cpu",
                onload_dtype=dtype,
                onload_device="cpu",
                computation_dtype=self.torch_dtype,
                computation_device=self.device,
            ),
        )
        self.enable_cpu_offload()

    def fetch_models(self, model_manager):
        text_encoder_model_and_path = model_manager.fetch_model("wan_video_text_encoder", require_model_path=True)
        if text_encoder_model_and_path is not None:
            self.text_encoder, tokenizer_path = text_encoder_model_and_path
            self.prompter.fetch_models(self.text_encoder)
            self.prompter.fetch_tokenizer(os.path.join(os.path.dirname(tokenizer_path), "google/umt5-xxl"))
        self.dit = model_manager.fetch_model("wan_video_dit")
        self.vae = model_manager.fetch_model("wan_video_vae")
        self.image_encoder = model_manager.fetch_model("wan_video_image_encoder")

    @staticmethod
    def from_model_manager(model_manager, torch_dtype=None, device=None, use_usp=False):
        if device is None:
            device = model_manager.device
        if torch_dtype is None:
            torch_dtype = model_manager.torch_dtype
        pipe = WanVideoPipeline(device=device, torch_dtype=torch_dtype)
        pipe.fetch_models(model_manager)
        assert not use_usp, "`use_usp` is not supported"
        return pipe

    def denoising_model(self):
        return self.dit

    def encode_prompt(self, prompt, positive=True):
        prompt_emb = self.prompter.encode_prompt(prompt, positive=positive, device=self.device)
        return {"context": prompt_emb}

    def encode_image(self, image: Union[torch.Tensor, Image.Image, np.ndarray], return_clip_feature=True, **kwargs):
        if not isinstance(image, torch.Tensor):
            image = torch.from_numpy(np.array(image)).unsqueeze(0)
            image = rearrange(image, "f h w c -> c f h w")
            image = (image / 255.0) * 2.0 - 1.0

        image = image.to(dtype=self.torch_dtype, device=self.device)
        image_latents = self.vae.encode([image], device=self.device, **kwargs)[0]  # [16, 1, 160, 90]
        image_latents = image_latents.to(self.device)

        if return_clip_feature:
            image = image.squeeze(1).unsqueeze(0)
            clip_feature = self.image_encoder.encode_image([image])  # [1, 257, 1280]
            return image_latents, clip_feature

        return image_latents

    def encode_cond(self, pose_frames=None, hand_frames=None, head_frames=None, tail_frames=None, **kwargs):
        assert pose_frames is not None or hand_frames is not None, "`pose_frames` or `hand_frames` should be provided"

        if pose_frames is not None:
            pose_frames = pose_frames.detach().clone()
        if hand_frames is not None:
            hand_frames = hand_frames.detach().clone()

        if isinstance(head_frames, (tuple, list)) and len(head_frames) == 0:
            head_frames = None
        if isinstance(tail_frames, (tuple, list)) and len(tail_frames) == 0:
            tail_frames = None

        head_length = 0
        tail_length = 0

        if head_frames is not None:
            if not isinstance(head_frames, torch.Tensor):  # for inference
                head_frames = [torch.from_numpy(np.array(frame)) for frame in head_frames]
                head_frames = rearrange(torch.stack(head_frames), "n h w c -> c n h w")
                head_frames = (head_frames / 255.0) * 2.0 - 1.0
            head_length = head_frames.shape[1]
            if pose_frames is not None:
                pose_frames[:, :head_length] = head_frames
            if hand_frames is not None:
                hand_frames[:, :head_length] = head_frames
        if tail_frames is not None:
            if not isinstance(tail_frames, torch.Tensor):  # for inference
                tail_frames = [torch.from_numpy(np.array(frame)) for frame in tail_frames]
                tail_frames = rearrange(torch.stack(tail_frames), "n h w c -> c n h w")
                tail_frames = (tail_frames / 255.0) * 2.0 - 1.0
            tail_length = tail_frames.shape[1]
            if pose_frames is not None:
                pose_frames[:, -tail_length:] = tail_frames
            if hand_frames is not None:
                hand_frames[:, -tail_length:] = tail_frames

        pose_latents = None
        if pose_frames is not None:
            pose_frames = pose_frames.to(dtype=self.torch_dtype, device=self.device)
            pose_latents = self.vae.encode([pose_frames], device=self.device, **kwargs)[0]  # [16, 21, 160, 90]
            pose_latents = pose_latents.to(dtype=self.torch_dtype, device=self.device)

        hand_latents = None
        if hand_frames is not None:
            hand_frames = hand_frames.to(dtype=self.torch_dtype, device=self.device)
            hand_latents = self.vae.encode([hand_frames], device=self.device, **kwargs)[0]  # [16, 21, 160, 90]
            hand_latents = hand_latents.to(dtype=self.torch_dtype, device=self.device)

        if pose_frames is not None:
            channels, num_frames, height, width = pose_frames.shape
            mask = torch.zeros(num_frames, height // 8, width // 8)
            if head_length > 0:
                mask[:head_length] = 1
            if tail_length > 0:
                mask[-tail_length:] = 1
            mask = torch.cat([torch.repeat_interleave(mask[:1], repeats=4, dim=0), mask[1:]], dim=0)
            mask = mask.reshape(mask.shape[0] // 4, 4, height // 8, width // 8)  # [21, 4, 160, 90]
            mask = mask.transpose(0, 1)  # [4, 21, 160, 90]
            mask = mask.to(dtype=self.torch_dtype, device=self.device)
            pose_latents = torch.cat([mask, pose_latents], dim=0)  # [20, 21, 160, 90]

        return pose_latents, hand_latents

    def tensor2video(self, frames):
        frames = rearrange(frames, "c f h w -> f h w c")
        frames = ((frames.float() + 1) * 127.5).clip(0, 255).cpu().numpy().astype(np.uint8)
        # frames = [Image.fromarray(frame) for frame in frames]
        frames = [frame for frame in frames]
        return frames

    def encode_video(self, input_video, tiled=True, tile_size=(34, 34), tile_stride=(18, 16)):
        latents = self.vae.encode(input_video, device=self.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)
        return latents

    def decode_video(self, latents, tiled=True, tile_size=(34, 34), tile_stride=(18, 16)):
        frames = self.vae.decode(latents, device=self.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)
        return frames

    def prepare_unified_sequence_parallel(self):
        return {"use_unified_sequence_parallel": self.use_unified_sequence_parallel}

    @torch.no_grad()
    def __call__(
        self,
        prompt,
        negative_prompt,
        input_image,
        input_pose_video,
        input_hand_video,
        head_frames=None,
        tail_frames=None,
        denoising_strength=1.0,
        seed=None,
        rand_device="cpu",
        height=1280,
        width=720,
        num_frames=81,
        cfg_scale=5.0,
        num_inference_steps=50,
        sigma_shift=5.0,
        tiled=True,
        tile_size=(30, 52),
        tile_stride=(15, 26),
        qkv_save_dir=None,
        qkv_load_dir=None,
        use_attn_mask=None,
        token_indices=None,
        max_block_idx=None,
        attn_maps_dir=None,
    ):
        height, width = self.check_resize_height_width(height, width)
        if num_frames % 4 != 1:
            num_frames = (num_frames + 2) // 4 * 4 + 1
            print(f"Only `num_frames % 4 != 1` is acceptable. We round it up to {num_frames}.")

        tiler_kwargs = {"tiled": tiled, "tile_size": tile_size, "tile_stride": tile_stride}

        self.scheduler.set_timesteps(num_inference_steps, denoising_strength=denoising_strength, shift=sigma_shift)

        noise = self.generate_noise(
            (1, 16, (num_frames - 1) // 4 + 1, height // 8, width // 8),
            seed=seed,
            device=rand_device,
            dtype=torch.float32,
        )
        noise = noise.to(dtype=self.torch_dtype, device=self.device)
        # if input_video is not None:
        #     self.load_models_to_device(["vae"])
        #     input_video = self.preprocess_images(input_video)
        #     input_video = torch.stack(input_video, dim=2).to(dtype=self.torch_dtype, device=self.device)
        #     latents = self.encode_video(input_video, **tiler_kwargs).to(dtype=self.torch_dtype, device=self.device)
        #     latents = self.scheduler.add_noise(latents, noise, timestep=self.scheduler.timesteps[0])
        # else:
        #     latents = noise
        latents = noise

        # Encode text prompt
        self.load_models_to_device(["text_encoder"])
        prompt_emb_posi = self.encode_prompt(prompt, positive=True)
        if cfg_scale != 1.0:
            prompt_emb_nega = self.encode_prompt(negative_prompt, positive=False)

        # Encode reference image
        self.load_models_to_device(["image_encoder", "vae"])
        image_latents, clip_feature = self.encode_image(input_image, return_clip_feature=True)
        image_latents = image_latents.unsqueeze(0)

        # Encode driving conditions
        self.load_models_to_device(["vae"])
        cond_latents, hand_latents = self.encode_cond(input_pose_video, input_hand_video, head_frames, tail_frames, **tiler_kwargs)
        cond_latents = cond_latents.unsqueeze(0)
        hand_latents = hand_latents.unsqueeze(0) if hand_latents is not None else None

        if use_attn_mask:
            assert attn_maps_dir is not None and attn_maps_dir != ""
        if attn_maps_dir:
            os.makedirs(attn_maps_dir, exist_ok=True)
        if qkv_save_dir:
            os.makedirs(os.path.join(qkv_save_dir, "sa_positive"), exist_ok=True)
            os.makedirs(os.path.join(qkv_save_dir, "sa_negative"), exist_ok=True)
            os.makedirs(os.path.join(qkv_save_dir, "ca_positive"), exist_ok=True)

        self.load_models_to_device(["dit"])
        for step_idx, timestep in enumerate(tqdm(self.scheduler.timesteps)):
            timestep = timestep.unsqueeze(0).to(dtype=self.torch_dtype, device=self.device)

            noise_pred_posi = self.dit(
                latents,
                image_latents,
                timestep=timestep,
                clip_feature=clip_feature,
                cond_latents=cond_latents,
                hand_latents=hand_latents,
                step_idx=step_idx,
                qkv_save_dir=qkv_save_dir,
                qkv_load_dir=qkv_load_dir,
                use_attn_mask=use_attn_mask,
                token_indices=token_indices,
                max_block_idx=max_block_idx,
                attn_maps_dir=attn_maps_dir,
                prompt_type="positive",
                **prompt_emb_posi,
            )

            if cfg_scale != 1.0:
                noise_pred_nega = self.dit(
                    latents,
                    image_latents,
                    timestep=timestep,
                    clip_feature=clip_feature,
                    cond_latents=cond_latents,
                    hand_latents=hand_latents,
                    step_idx=step_idx,
                    qkv_save_dir=qkv_save_dir,
                    qkv_load_dir=qkv_load_dir,
                    use_attn_mask=use_attn_mask,
                    token_indices=token_indices,
                    max_block_idx=max_block_idx,
                    attn_maps_dir=attn_maps_dir,
                    prompt_type="negative",
                    **prompt_emb_nega,
                )
                noise_pred = noise_pred_nega + cfg_scale * (noise_pred_posi - noise_pred_nega)
            else:
                noise_pred = noise_pred_posi

            latents = self.scheduler.step(noise_pred, self.scheduler.timesteps[step_idx], latents)

        self.load_models_to_device(["vae"])
        frames = self.decode_video(latents, **tiler_kwargs)
        self.load_models_to_device([])
        frames = self.tensor2video(frames[0])

        return frames
