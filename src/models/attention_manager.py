import math
import os
from typing import Union

from einops import rearrange
import matplotlib.pyplot as plt
import numpy as np
import torch


class AttentionManager:
    def __init__(self, num_heads=40):
        self.num_heads = num_heads
        self.attn_map_sum = None
        self.attn_map_cnt = 0

    def reset(self):
        self.attn_map_sum = None
        self.attn_map_cnt = 0

    def save_attn_map(self, q, k):
        q = rearrange(q, "b s (n d) -> b n s d", n=self.num_heads)  # [1, 75600, 5120] -> [1, 40, 75600, 128]
        k = rearrange(k, "b s (n d) -> b n s d", n=self.num_heads)  # [1, 512, 5120] -> [1, 40, 512, 128]

        batch_size, num_heads, q_seq_len, head_dim = q.shape
        batch_size, num_heads, k_seq_len, head_dim = k.shape

        # Compute attention map
        attn_map = torch.zeros((batch_size, q_seq_len, k_seq_len), device=q.device, dtype=q.dtype)
        for i in range(batch_size):
            for j in range(num_heads):
                q_chunk = q[i : i + 1, j : j + 1]
                k_chunk = k[i : i + 1, j : j + 1]
                attn = torch.matmul(q_chunk, k_chunk.transpose(-1, -2)) / math.sqrt(head_dim)  # [1, 1, 75600, 512]
                attn = torch.softmax(attn, dim=-1)
                attn = attn.sum(dim=1)  # [1, 75600, 512]
                attn_map[i : i + 1] += attn
        attn_map /= num_heads

        if self.attn_map_sum is None:
            self.attn_map_sum = attn_map
        else:
            self.attn_map_sum += attn_map
        self.attn_map_cnt += 1

    def _get_attn_map(self, f, h, w, token_idx: Union[int, list[int], tuple[int, ...]]):
        if self.attn_map_sum is None or self.attn_map_cnt == 0:
            return None

        attn_map = self.attn_map_sum / self.attn_map_cnt  # [b fhw t]
        attn_map = attn_map.reshape(attn_map.shape[0], f, h, w, attn_map.shape[-1])

        if isinstance(token_idx, (list, tuple)):
            attn_map = attn_map[..., token_idx].sum(-1)
        else:
            attn_map = attn_map[..., token_idx]

        min_value = attn_map.amin((-2, -1), keepdim=True)
        max_value = attn_map.amax((-2, -1), keepdim=True)
        attn_map = (attn_map - min_value) / (max_value - min_value + 1e-6)
        return attn_map

    def get_attn_map(self, f, h, w, token_idx, frame_idx):
        if isinstance(frame_idx, (list, tuple)):
            one_frame_idx = False
        else:
            one_frame_idx = True
            frame_idx = [frame_idx]

        attn_maps = []
        for fi in frame_idx:
            # if layout == "horizontal":
            #     fig, axes = plt.subplots(1, num_tokens, figsize=(2.0 * num_tokens, 2))
            #     subplot_adjust = {"wspace": 0.05, "hspace": 0}
            # else:
            #     fig, axes = plt.subplots(num_tokens, 1, figsize=(2, 1.2 * num_tokens))
            #     subplot_adjust = {"wspace": 0, "hspace": 0.01}

            # if num_tokens == 1:
            #     axes = [axes]

            # for idx, token_idx in enumerate(token_indices):
            attn_map = self._get_attn_map(f=f, h=h, w=w, token_idx=token_idx)  # [b f h w]
            attn_map = attn_map[0, fi, :, :]  # [h w]

            attn_map = plt.get_cmap("magma")(attn_map.float().cpu())
            attn_map = (attn_map[:, :, :3] * 255).astype(np.uint8)
            attn_maps.append(attn_map)

            # axes[idx].imshow(img)
            # axes[idx].axis("off")

            # plt.subplots_adjust(**subplot_adjust)

            # if layout == "vertical":
            #     plt.tight_layout(pad=0.1, h_pad=0.1)

            # os.makedirs(save_dir, exist_ok=True)
            # save_path = os.path.join(save_dir, f"timestep{timestep_idx}_block{block_idx}_frame{frame_idx}.png")
            # plt.savefig(save_path, dpi=150, bbox_inches="tight", pad_inches=0)
            # plt.close()

        if one_frame_idx:
            return attn_maps[0]
        else:
            return attn_maps
