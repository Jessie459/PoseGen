import math
import os
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from .utils import hash_state_dict_keys, init_weights_on_device
from .thresholding_utils import otsu_threshold

# try:
#     import flash_attn_interface
#     FLASH_ATTN_3_AVAILABLE = True
# except ModuleNotFoundError:
#     FLASH_ATTN_3_AVAILABLE = False
# try:
#     import flash_attn
#     FLASH_ATTN_2_AVAILABLE = True
# except ModuleNotFoundError:
#     FLASH_ATTN_2_AVAILABLE = False
# try:
#     from sageattention import sageattn
#     SAGE_ATTN_AVAILABLE = True
# except ModuleNotFoundError:
#     SAGE_ATTN_AVAILABLE = False


# def flash_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, num_heads: int, compatibility_mode=False):
#     if compatibility_mode:
#         q = rearrange(q, "b s (n d) -> b n s d", n=num_heads)
#         k = rearrange(k, "b s (n d) -> b n s d", n=num_heads)
#         v = rearrange(v, "b s (n d) -> b n s d", n=num_heads)
#         x = F.scaled_dot_product_attention(q, k, v)
#         x = rearrange(x, "b n s d -> b s (n d)", n=num_heads)
#     elif FLASH_ATTN_3_AVAILABLE:
#         q = rearrange(q, "b s (n d) -> b s n d", n=num_heads)
#         k = rearrange(k, "b s (n d) -> b s n d", n=num_heads)
#         v = rearrange(v, "b s (n d) -> b s n d", n=num_heads)
#         x = flash_attn_interface.flash_attn_func(q, k, v)
#         if isinstance(x, tuple):
#             x = x[0]
#         x = rearrange(x, "b s n d -> b s (n d)", n=num_heads)
#     elif FLASH_ATTN_2_AVAILABLE:
#         q = rearrange(q, "b s (n d) -> b s n d", n=num_heads)
#         k = rearrange(k, "b s (n d) -> b s n d", n=num_heads)
#         v = rearrange(v, "b s (n d) -> b s n d", n=num_heads)
#         x = flash_attn.flash_attn_func(q, k, v)
#         x = rearrange(x, "b s n d -> b s (n d)", n=num_heads)
#     elif SAGE_ATTN_AVAILABLE:
#         q = rearrange(q, "b s (n d) -> b n s d", n=num_heads)
#         k = rearrange(k, "b s (n d) -> b n s d", n=num_heads)
#         v = rearrange(v, "b s (n d) -> b n s d", n=num_heads)
#         x = sageattn(q, k, v)
#         x = rearrange(x, "b n s d -> b s (n d)", n=num_heads)
#     else:
#         q = rearrange(q, "b s (n d) -> b n s d", n=num_heads)
#         k = rearrange(k, "b s (n d) -> b n s d", n=num_heads)
#         v = rearrange(v, "b s (n d) -> b n s d", n=num_heads)
#         x = F.scaled_dot_product_attention(q, k, v)
#         x = rearrange(x, "b n s d -> b s (n d)", n=num_heads)
#     return x


def compute_attn_map(q: torch.Tensor, k: torch.Tensor, num_heads=40):
    q = rearrange(q, "b s (n d) -> b n s d", n=num_heads)
    k = rearrange(k, "b s (n d) -> b n s d", n=num_heads)
    head_dim = q.shape[-1]
    attn_map = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(head_dim)
    attn_map = torch.mean(attn_map, dim=1)
    return attn_map


def normalize_attn_map(attn_map: torch.Tensor, eps=1e-6):
    min_value = attn_map.amin((-2, -1), keepdim=True)
    max_value = attn_map.amax((-2, -1), keepdim=True)
    attn_map = (attn_map - min_value) / (max_value - min_value + eps)
    return attn_map


def flash_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, num_heads: int, attn_mask: Optional[torch.Tensor] = None):
    q = rearrange(q, "b s (n d) -> b n s d", n=num_heads)
    k = rearrange(k, "b s (n d) -> b n s d", n=num_heads)
    v = rearrange(v, "b s (n d) -> b n s d", n=num_heads)
    x = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
    x = rearrange(x, "b n s d -> b s (n d)", n=num_heads)
    return x


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor):
    return x * (1 + scale) + shift


def sinusoidal_embedding_1d(dim, position):
    sinusoid = torch.outer(
        position.type(torch.float64),
        torch.pow(10000, -torch.arange(dim // 2, dtype=torch.float64, device=position.device).div(dim // 2)),
    )
    x = torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1)
    return x.to(position.dtype)


def precompute_freqs_cis_3d(dim: int, end: int = 1024, theta: float = 10000.0):
    # 3d rope precompute
    f_freqs_cis = precompute_freqs_cis(dim - 2 * (dim // 3), end, theta)
    h_freqs_cis = precompute_freqs_cis(dim // 3, end, theta)
    w_freqs_cis = precompute_freqs_cis(dim // 3, end, theta)
    return f_freqs_cis, h_freqs_cis, w_freqs_cis


def precompute_freqs_cis(dim: int, end: int = 1024, theta: float = 10000.0, delta: Optional[int] = None):
    # 1d rope precompute
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].double() / dim))
    if delta is None:
        freqs = torch.outer(torch.arange(end, device=freqs.device), freqs)
    else:
        freqs = torch.outer(torch.arange(end, device=freqs.device) + delta, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis


def rope_apply(x, freqs, num_heads):
    x = rearrange(x, "b s (n d) -> b s n d", n=num_heads)
    x_out = torch.view_as_complex(x.to(torch.float64).reshape(x.shape[0], x.shape[1], x.shape[2], -1, 2))
    x_out = torch.view_as_real(x_out * freqs).flatten(2)
    return x_out.to(x.dtype)


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)

    def forward(self, x):
        dtype = x.dtype
        return self.norm(x.float()).to(dtype) * self.weight


class AttentionModule(nn.Module):
    def __init__(self, num_heads):
        super().__init__()
        self.num_heads = num_heads

    def forward(self, q, k, v, attn_mask=None):
        x = flash_attention(q=q, k=k, v=v, num_heads=self.num_heads, attn_mask=attn_mask)
        return x


class SelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = RMSNorm(dim, eps=eps)
        self.norm_k = RMSNorm(dim, eps=eps)

        self.attn = AttentionModule(self.num_heads)

    def forward(self, x, freqs, ref, ref_freqs):
        q = self.norm_q(self.q(x))
        k = self.norm_k(self.k(x))
        v = self.v(x)
        q = rope_apply(q, freqs, self.num_heads)  # [1, 75600, 5120]
        k = rope_apply(k, freqs, self.num_heads)

        ref_q = self.norm_q(self.q(ref))
        ref_k = self.norm_k(self.k(ref))
        ref_v = self.v(ref)
        ref_q = rope_apply(ref_q, ref_freqs, self.num_heads)
        ref_k = rope_apply(ref_k, ref_freqs, self.num_heads)

        q = torch.cat([q, ref_q], dim=1)  # [1, 75600+3600, 5120]
        k = torch.cat([k, ref_k], dim=1)
        v = torch.cat([v, ref_v], dim=1)

        attn = self.attn(q, k, v)
        x, ref = attn[:, :x.shape[1]], attn[:, x.shape[1]:]

        x = self.o(x)
        ref = self.o(ref)

        return x, ref


class MaskedKVShareSelfAttention(SelfAttention):
    def __init__(self, dim: int, num_heads: int, eps: float = 1e-6):
        super().__init__(dim, num_heads, eps)

    def forward(self, x, freqs, ref, ref_freqs, kvshare, source_k, source_v, attn_map, qkv_save_dir, step_idx, block_idx, is_anch, is_base):
        b, seq_len, _ = x.shape
        b, ref_seq_len, _ = ref.shape

        q = self.norm_q(self.q(x))
        k = self.norm_k(self.k(x))
        v = self.v(x)
        q = rope_apply(q, freqs, self.num_heads)
        k = rope_apply(k, freqs, self.num_heads)

        ref_q = self.norm_q(self.q(ref))
        ref_k = self.norm_k(self.k(ref))
        ref_v = self.v(ref)
        ref_q = rope_apply(ref_q, ref_freqs, self.num_heads)
        ref_k = rope_apply(ref_k, ref_freqs, self.num_heads)

        q = torch.cat([q, ref_q], dim=1)
        k = torch.cat([k, ref_k], dim=1)
        v = torch.cat([v, ref_v], dim=1)

        if kvshare and is_anch:
            torch.save(k.detach().cpu(), os.path.join(qkv_save_dir, f"step{step_idx}_block{block_idx}_k.pt"))
            torch.save(v.detach().cpu(), os.path.join(qkv_save_dir, f"step{step_idx}_block{block_idx}_v.pt"))

        attn = self.attn(q, k, v)
        x, ref = attn[:, :seq_len], attn[:, seq_len:]
        x, ref = self.o(x), self.o(ref)

        x2, ref2 = None, None
        if kvshare and is_base:
            if attn_map is not None:
                attn_mask = torch.zeros((b, seq_len + ref_seq_len, seq_len + ref_seq_len), dtype=q.dtype)
                for i in range(b):
                    indices = (attn_map[i] == 1).nonzero(as_tuple=False).squeeze(-1).cpu()
                    if len(indices) > 0:
                        attn_mask[i, indices[:, None], indices[None, :]] = -torch.inf
                attn2 = self.attn(q, source_k, source_v, attn_mask=attn_mask.to(q.device))
            else:
                attn2 = self.attn(q, source_k, source_v)
            x2, ref2 = attn2[:, :seq_len], attn2[:, seq_len:]
            x2, ref2 = self.o(x2), self.o(ref2) 

        return x, ref, x2, ref2


class CrossAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, eps: float = 1e-6, has_image_input: bool = False):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = RMSNorm(dim, eps=eps)
        self.norm_k = RMSNorm(dim, eps=eps)
        self.has_image_input = has_image_input
        if has_image_input:
            self.k_img = nn.Linear(dim, dim)
            self.v_img = nn.Linear(dim, dim)
            self.norm_k_img = RMSNorm(dim, eps=eps)

        self.attn = AttentionModule(self.num_heads)

    def forward(self, x, y):
        if self.has_image_input:
            img = y[:, :257]  # [1, 257, 5120]
            ctx = y[:, 257:]  # [1, 512, 5120]
        else:
            ctx = y
        q = self.norm_q(self.q(x))  # [1, 75600, 5120]
        k = self.norm_k(self.k(ctx))  # [1, 512, 5120]
        v = self.v(ctx)  # [1, 512, 5120]

        x = self.attn(q, k, v)
        if self.has_image_input:
            k_img = self.norm_k_img(self.k_img(img))
            v_img = self.v_img(img)
            y = flash_attention(q, k_img, v_img, num_heads=self.num_heads)
            x = x + y
        return self.o(x)


class MaskedKVShareCrossAttention(CrossAttention):
    def __init__(self, dim: int, num_heads: int, eps: float = 1e-6, has_image_input: bool = False):
        super().__init__(dim, num_heads, eps, has_image_input)

    def forward(self, x, y, use_attn_mask, token_indices, attn_map_size, prompt_type, qkv_save_dir, step_idx, block_idx, kvshare_this_step, is_anch, is_base):
        if self.has_image_input:
            img = y[:, :257]
            ctx = y[:, 257:]
        else:
            ctx = y

        q = self.norm_q(self.q(x))
        k = self.norm_k(self.k(ctx))
        v = self.v(ctx)

        attn_map = None
        if kvshare_this_step:
            if is_anch:
                if prompt_type == "positive":
                    torch.save(q.detach().cpu(), os.path.join(qkv_save_dir, f"step{step_idx}_block{block_idx}_q.pt"))
                    torch.save(k.detach().cpu(), os.path.join(qkv_save_dir, f"step{step_idx}_block{block_idx}_k.pt"))
            if is_base:
                if use_attn_mask:
                    attn_map = compute_attn_map(q, k, num_heads=self.num_heads)
                    attn_map = attn_map[:, :, token_indices].sum(-1)
                    attn_map = attn_map.reshape(q.shape[0], *attn_map_size)
                    attn_map = normalize_attn_map(attn_map)

        x = self.attn(q, k, v)

        if self.has_image_input:
            k_img = self.norm_k_img(self.k_img(img))
            v_img = self.v_img(img)
            y = flash_attention(q, k_img, v_img, num_heads=self.num_heads)
            x = x + y

        return self.o(x), attn_map


class GateModule(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, gate, residual):
        return x + gate * residual


class DiTBlock(nn.Module):
    def __init__(self, has_image_input: bool, dim: int, num_heads: int, ffn_dim: int, eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.ffn_dim = ffn_dim

        self.self_attn = MaskedKVShareSelfAttention(dim, num_heads, eps)
        self.cross_attn = MaskedKVShareCrossAttention(dim, num_heads, eps, has_image_input=has_image_input)
        self.norm1 = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
        self.norm3 = nn.LayerNorm(dim, eps=eps)
        self.ffn = nn.Sequential(nn.Linear(dim, ffn_dim), nn.GELU(approximate="tanh"), nn.Linear(ffn_dim, dim))
        self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)
        self.gate = GateModule()

    def forward(
        self,
        x,
        context,
        t_mod,
        freqs,
        ref,
        ref_t_mod,
        ref_freqs,
        is_anch=None,
        is_base=None,
        kvshare=None,
        kvshare_this_step=None,
        source_k=None,
        source_v=None,
        use_attn_mask=None,
        token_indices=None,
        prev_attn_map_sum=None,
        prev_attn_map_cnt=None,
        curr_attn_map_sum=None,
        curr_attn_map_cnt=None,
        attn_map_size=None,
        prompt_type=None,
        qkv_save_dir=None,
        step_idx=None,
        block_idx=None,
    ):
        if t_mod.dim() == 3:  # [b 6 dim]
            modulation = self.modulation
            out = (modulation.to(t_mod) + t_mod).chunk(6, dim=1)
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = out  # [b 1 dim]
        else:  # [b 6 n dim]
            modulation = self.modulation.unsqueeze(2)
            out = (modulation.to(t_mod) + t_mod).chunk(6, dim=1)
            out = [o.squeeze(1) for o in out]
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = out  # [b n dim]

        # shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
        #     modulation + t_mod
        # ).chunk(6, dim=1)

        ref_modulation = self.modulation
        out = (ref_modulation.to(ref_t_mod) + ref_t_mod).chunk(6, dim=1)
        ref_shift_msa, ref_scale_msa, ref_gate_msa, ref_shift_mlp, ref_scale_mlp, ref_gate_mlp = out

        input_x = modulate(self.norm1(x), shift_msa, scale_msa)
        input_ref = modulate(self.norm1(ref), ref_shift_msa, ref_scale_msa)

        prev_attn_map = None
        curr_attn_map = None
        if kvshare and is_base:
            if use_attn_mask:
                batch_size = x.shape[0]
                prev_attn_map = prev_attn_map_sum / prev_attn_map_cnt
                prev_attn_map = otsu_threshold(rearrange(prev_attn_map, "b f h w -> (b f) h w"))
                prev_attn_map = rearrange(prev_attn_map, "(b f) h w -> b (f h w)", b=batch_size)
                if curr_attn_map_sum is not None:  # the first block does not have current attention maps
                    curr_attn_map = curr_attn_map_sum / curr_attn_map_cnt
                    curr_attn_map = otsu_threshold(rearrange(curr_attn_map, "b f h w -> (b f) h w"))
                    curr_attn_map = rearrange(curr_attn_map, "(b f) h w -> b (f h w)", b=batch_size)

        out_x, out_ref, out_x2, out_ref2 = self.self_attn(
            input_x,
            freqs,
            input_ref,
            ref_freqs,
            kvshare=kvshare,
            source_k=source_k,
            source_v=source_v,
            attn_map=prev_attn_map,
            qkv_save_dir=os.path.join(qkv_save_dir, f"sa_{prompt_type}") if qkv_save_dir else None,
            step_idx=step_idx,
            block_idx=block_idx,
            is_anch=is_anch,
            is_base=is_base,
        )

        x2 = self.gate(x, gate_msa, out_x2) if out_x2 is not None else None
        ref2 = self.gate(ref, ref_gate_msa, out_ref2) if out_ref2 is not None else None

        x = self.gate(x, gate_msa, out_x)
        ref = self.gate(ref, ref_gate_msa, out_ref)

        if kvshare and is_base:  # x2 and ref2 are not None
            if use_attn_mask:
                _attn_map = torch.bitwise_or(curr_attn_map, prev_attn_map)
                x[_attn_map == 0] = x2[_attn_map == 0]
            else:
                x = x2
            ref = ref2

        out_x, attn_map = self.cross_attn(
            self.norm3(x),
            context,
            use_attn_mask=use_attn_mask,
            token_indices=token_indices,
            attn_map_size=attn_map_size,
            prompt_type=prompt_type,
            qkv_save_dir=os.path.join(qkv_save_dir, f"ca_{prompt_type}") if qkv_save_dir else None,
            step_idx=step_idx,
            block_idx=block_idx,
            kvshare_this_step=kvshare_this_step,
            is_anch=is_anch,
            is_base=is_base,
        )
        x = x + out_x

        input_x = modulate(self.norm2(x), shift_mlp, scale_mlp)
        input_ref = modulate(self.norm2(ref), ref_shift_mlp, ref_scale_mlp)
        x = self.gate(x, gate_mlp, self.ffn(input_x))
        ref = self.gate(ref, ref_gate_mlp, self.ffn(input_ref))

        return x, ref, attn_map


class MLP(torch.nn.Module):
    def __init__(self, in_dim, out_dim, has_pos_emb=False):
        super().__init__()
        self.proj = torch.nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, in_dim),
            nn.GELU(),
            nn.Linear(in_dim, out_dim),
            nn.LayerNorm(out_dim),
        )
        self.has_pos_emb = has_pos_emb
        if has_pos_emb:
            self.emb_pos = torch.nn.Parameter(torch.zeros((1, 514, 1280)))

    def forward(self, x):
        if self.has_pos_emb:
            x = x + self.emb_pos.to(dtype=x.dtype, device=x.device)
        return self.proj(x)


class Head(nn.Module):
    def __init__(self, dim: int, out_dim: int, patch_size: Tuple[int, int, int], eps: float):
        super().__init__()
        self.dim = dim
        self.patch_size = patch_size
        self.norm = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
        self.head = nn.Linear(dim, out_dim * math.prod(patch_size))
        self.modulation = nn.Parameter(torch.randn(1, 2, dim) / dim**0.5)

    def forward(self, x, t_mod):
        if t_mod.dim() == 2:  # [1, 5120]
            modulation = self.modulation  # [1, 2, 5120]
            shift, scale = (modulation.to(t_mod) + t_mod.unsqueeze(1)).chunk(2, dim=1)
        else:  # [1, fhw, 5120]
            modulation = self.modulation.unsqueeze(2)  # [1, 2, 1, 5120]
            shift, scale = (modulation.to(t_mod) + t_mod.unsqueeze(1)).chunk(2, dim=1)
            shift, scale = shift.squeeze(1), scale.squeeze(1)

        # shift, scale = (self.modulation.to(dtype=t_mod.dtype, device=t_mod.device) + t_mod).chunk(2, dim=1)

        x = self.head(self.norm(x) * (1 + scale) + shift)
        return x


class WanModel(torch.nn.Module):
    def __init__(
        self,
        dim: int,
        in_dim: int,
        ffn_dim: int,
        out_dim: int,
        text_dim: int,
        freq_dim: int,
        eps: float,
        patch_size: Tuple[int, int, int],
        num_heads: int,
        num_layers: int,
        has_image_input: bool,
        has_image_pos_emb: bool = False,
        shift_ref_pos: bool = False,
        use_hand: bool = False,
        use_hand_proj: bool = False,
        kvshare_kwargs: dict = None,
    ):
        super().__init__()
        self.dim = dim
        self.freq_dim = freq_dim
        self.has_image_input = has_image_input
        self.patch_size = patch_size
        self.num_heads = num_heads

        self.patch_embedding = nn.Conv3d(in_dim, dim, kernel_size=patch_size, stride=patch_size)
        self.ref_patch_embedding = nn.Conv3d(16, dim, kernel_size=patch_size, stride=patch_size)
        self.use_hand = use_hand
        self.use_hand_proj = use_hand_proj
        if self.use_hand:
            if self.use_hand_proj:
                self.hand_patch_embedding = nn.Sequential(
                    nn.Conv3d(16, dim, kernel_size=patch_size, stride=patch_size),
                    nn.Conv3d(dim, dim, kernel_size=1),
                )
            else:
                self.hand_patch_embedding = nn.Conv3d(16, dim, kernel_size=patch_size, stride=patch_size)
        self.text_embedding = nn.Sequential(nn.Linear(text_dim, dim), nn.GELU(approximate="tanh"), nn.Linear(dim, dim))
        self.time_embedding = nn.Sequential(nn.Linear(freq_dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.time_projection = nn.Sequential(nn.SiLU(), nn.Linear(dim, dim * 6))
        self.blocks = nn.ModuleList([DiTBlock(has_image_input, dim, num_heads, ffn_dim, eps) for _ in range(num_layers)])
        self.head = Head(dim, out_dim, patch_size, eps)
        head_dim = dim // num_heads
        self.freqs = precompute_freqs_cis_3d(head_dim)

        self.shift_ref_pos = shift_ref_pos
        ref_freqs_dims = (head_dim - 2 * (head_dim // 3), head_dim // 3, head_dim // 3)
        ref_freqs_f = precompute_freqs_cis(ref_freqs_dims[0], 1024)
        ref_freqs_h = precompute_freqs_cis(ref_freqs_dims[1], 1024)
        ref_freqs_w = precompute_freqs_cis(ref_freqs_dims[2], 1024, delta=-1024 if self.shift_ref_pos else None)
        self.ref_freqs = (ref_freqs_f, ref_freqs_h, ref_freqs_w)

        if has_image_input:
            self.img_emb = MLP(1280, dim, has_pos_emb=has_image_pos_emb)  # clip_feature_dim = 1280
        self.has_image_pos_emb = has_image_pos_emb

        if kvshare_kwargs is None:
            kvshare_kwargs = {}
        self.kvshare_steps = kvshare_kwargs.get("steps", [])
        self.kvshare_blocks = kvshare_kwargs.get("blocks", [])
        print(f"kvshare steps: {self.kvshare_steps}")
        print(f"kvshare blocks: {self.kvshare_blocks}")
        self.prev_attn_map_sum = None
        self.prev_attn_map_cnt = 0
        self.curr_attn_map_sum = None
        self.curr_attn_map_cnt = 0

    def patchify(self, x: torch.Tensor, cond_latents: torch.Tensor, hand_latents: torch.Tensor = None):
        x = torch.cat([x, cond_latents], dim=1)  # [1, 16+20=36, 21, 160, 90]
        x = self.patch_embedding(x)  # [b 36 f h w] -> [b 5120 f h/2 w/2]
        if hand_latents is not None:
            hand_latents = self.hand_patch_embedding(hand_latents)  # [b 16 f h w] -> [b 5120 f h/2 w/2]
            x = x + hand_latents
        grid_size = x.shape[2:]
        x = rearrange(x, "b c f h w -> b (f h w) c").contiguous()
        return x, grid_size  # x, grid_size: (f, h, w)

    def ref_patchify(self, x: torch.Tensor):
        x = self.ref_patch_embedding(x)
        grid_size = x.shape[2:]
        x = rearrange(x, "b c f h w -> b (f h w) c").contiguous()
        return x, grid_size  # x, grid_size: (f, h, w)

    def unpatchify(self, x: torch.Tensor, grid_size: torch.Tensor):
        return rearrange(
            x,
            "b (f h w) (x y z c) -> b c (f x) (h y) (w z)",
            f=grid_size[0],
            h=grid_size[1],
            w=grid_size[2],
            x=self.patch_size[0],
            y=self.patch_size[1],
            z=self.patch_size[2],
        )

    def forward(
        self,
        x: torch.Tensor,
        ref: torch.Tensor,
        timestep: torch.Tensor,
        context: torch.Tensor,
        clip_feature: Optional[torch.Tensor] = None,
        cond_latents: Optional[torch.Tensor] = None,
        hand_latents: Optional[torch.Tensor] = None,
        use_gradient_checkpointing: bool = False,
        use_gradient_checkpointing_offload: bool = False,
        step_idx: int = None,
        qkv_save_dir: str = None,
        qkv_load_dir: str = None,
        use_attn_mask: bool = None,
        token_indices: list[int] = None,
        max_block_idx: int = None,
        attn_maps_dir: str = None,
        prompt_type: str = None,
        **kwargs,
    ):
        is_anch = qkv_save_dir is not None and qkv_save_dir != ""
        is_base = qkv_load_dir is not None and qkv_load_dir != ""
        if is_anch and is_base:
            raise RuntimeError("`is_anch` and `is_base` are both True")
        assert prompt_type in ["positive", "negative"]

        sa_load_dir, ca_load_dir = None, None
        if qkv_load_dir:
            sa_load_dir = os.path.join(qkv_load_dir, f"sa_{prompt_type}")
            ca_load_dir = os.path.join(qkv_load_dir, f"ca_positive")  # only calculate attention maps when prompts are positive

        if timestep.dim() == 1:
            _flag_df = False
            t = self.time_embedding(sinusoidal_embedding_1d(self.freq_dim, timestep))
            t_mod = self.time_projection(t).unflatten(1, (6, self.dim))
        elif timestep.dim() == 2:
            _flag_df = True
            t = self.time_embedding(sinusoidal_embedding_1d(self.freq_dim, timestep.view(-1)))
            t_mod = self.time_projection(t).unflatten(1, (6, self.dim))

        # t = self.time_embedding(sinusoidal_embedding_1d(self.freq_dim, timestep))
        # t_mod = self.time_projection(t).unflatten(1, (6, self.dim))

        ref_timestep = torch.zeros((timestep.shape[0],)).to(timestep)
        ref_t = self.time_embedding(sinusoidal_embedding_1d(self.freq_dim, ref_timestep))
        ref_t_mod = self.time_projection(ref_t).unflatten(1, (6, self.dim))

        context = self.text_embedding(context)  # [1, 512, 4096] -> [1, 512, 5120]

        assert self.has_image_input
        clip_embdding = self.img_emb(clip_feature)  # [1, 257, 1280] -> [1, 257, 5120]
        context = torch.cat([clip_embdding, context], dim=1)

        # x = torch.cat([x, y], dim=1)
        x, (f, h, w) = self.patchify(x, cond_latents, hand_latents)
        ref, (ref_f, ref_h, ref_w) = self.ref_patchify(ref)

        freqs = torch.cat(
            [
                self.freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
                self.freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
                self.freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1),
            ],
            dim=-1,
        )
        freqs = freqs.reshape(f * h * w, 1, -1).to(x.device)

        if not self.shift_ref_pos:
            ref_freqs = torch.cat(
                [
                    self.ref_freqs[0][:ref_f].view(ref_f, 1, 1, -1).expand(ref_f, ref_h, ref_w, -1),
                    self.ref_freqs[1][:ref_h].view(1, ref_h, 1, -1).expand(ref_f, ref_h, ref_w, -1),
                    self.ref_freqs[2][:ref_w].view(1, 1, ref_w, -1).expand(ref_f, ref_h, ref_w, -1),
                ],
                dim=-1,
            )
        else:
            ref_freqs = torch.cat(
                [
                    self.ref_freqs[0][:ref_f].view(ref_f, 1, 1, -1).expand(ref_f, ref_h, ref_w, -1),
                    self.ref_freqs[1][:ref_h].view(1, ref_h, 1, -1).expand(ref_f, ref_h, ref_w, -1),
                    self.ref_freqs[2][-ref_w:].view(1, 1, ref_w, -1).expand(ref_f, ref_h, ref_w, -1),
                ],
                dim=-1,
            )
        ref_freqs = ref_freqs.reshape(ref_f * ref_h * ref_w, 1, -1).to(x.device)

        if _flag_df:
            b = timestep.shape[0]
            t = t.view(b, f, 1, 1, self.dim).repeat(1, 1, h, w, 1).flatten(1, 3)  # [b (f h w) c]
            t_mod = t_mod.view(b, f, 1, 1, 6, self.dim).repeat(1, 1, h, w, 1, 1).flatten(1, 3)  # [b (f h w) 6 c]
            t_mod = t_mod.transpose(1, 2).contiguous()  # [b 6 (f h w) c]

        def create_custom_forward(module):
            def custom_forward(*inputs):
                return module(*inputs)

            return custom_forward

        if use_attn_mask:
            if max_block_idx is None:
                max_block_idx = len(self.blocks) - 1

        for block_idx, block in enumerate(self.blocks):
            if self.training and use_gradient_checkpointing:
                if use_gradient_checkpointing_offload:
                    with torch.autograd.graph.save_on_cpu():
                        x, ref = torch.utils.checkpoint.checkpoint(
                            create_custom_forward(block),
                            x,
                            context,
                            t_mod,
                            freqs,
                            ref,
                            ref_t_mod,
                            ref_freqs,
                            use_reentrant=False,
                        )
                else:
                    x, ref = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(block),
                        x,
                        context,
                        t_mod,
                        freqs,
                        ref,
                        ref_t_mod,
                        ref_freqs,
                        use_reentrant=False,
                    )
            else:
                kvshare_this_step = step_idx in self.kvshare_steps
                kvshare_this_block = block_idx in self.kvshare_blocks
                kvshare = kvshare_this_step and kvshare_this_block

                def _load_tensor(path):
                    return torch.load(path, weights_only=True, map_location="cpu")

                if kvshare_this_step and is_base:  # accumulate attention maps
                    if use_attn_mask and block_idx <= max_block_idx:
                        q = _load_tensor(os.path.join(ca_load_dir, f"step{step_idx}_block{block_idx}_q.pt")).to(x)
                        k = _load_tensor(os.path.join(ca_load_dir, f"step{step_idx}_block{block_idx}_k.pt")).to(x)
                        prev_attn_map = compute_attn_map(q, k, num_heads=self.num_heads)
                        prev_attn_map = prev_attn_map[..., token_indices].sum(-1)
                        prev_attn_map = prev_attn_map.reshape(x.shape[0], f, h, w)
                        prev_attn_map = normalize_attn_map(prev_attn_map)
                        if self.prev_attn_map_sum is None:
                            self.prev_attn_map_sum = prev_attn_map
                        else:
                            self.prev_attn_map_sum += prev_attn_map
                        self.prev_attn_map_cnt += 1

                source_k = None
                source_v = None
                if kvshare and is_base:
                    source_k = _load_tensor(os.path.join(sa_load_dir, f"step{step_idx}_block{block_idx}_k.pt")).to(x)
                    source_v = _load_tensor(os.path.join(sa_load_dir, f"step{step_idx}_block{block_idx}_v.pt")).to(x)

                x, ref, curr_attn_map = block(
                    x,
                    context,
                    t_mod,
                    freqs,
                    ref,
                    ref_t_mod,
                    ref_freqs,
                    is_anch=is_anch,
                    is_base=is_base,
                    kvshare=kvshare,
                    kvshare_this_step=kvshare_this_step,
                    source_k=source_k,
                    source_v=source_v,
                    use_attn_mask=use_attn_mask,
                    token_indices=token_indices,
                    prev_attn_map_sum=self.prev_attn_map_sum,
                    prev_attn_map_cnt=self.prev_attn_map_cnt,
                    curr_attn_map_sum=self.curr_attn_map_sum,
                    curr_attn_map_cnt=self.curr_attn_map_cnt,
                    attn_map_size=(f, h, w),
                    prompt_type=prompt_type,
                    qkv_save_dir=qkv_save_dir,
                    step_idx=step_idx,
                    block_idx=block_idx,
                )

                if kvshare_this_step and is_base:  # accumulate attention maps
                    if use_attn_mask and block_idx <= max_block_idx:
                        attn_maps_path = os.path.join(attn_maps_dir, f"step{step_idx}_block{block_idx}.pt")
                        if prompt_type == "positive":
                            torch.save(curr_attn_map.cpu(), attn_maps_path)
                        else:
                            curr_attn_map = _load_tensor(attn_maps_path).to(x)
                        if self.curr_attn_map_sum is None:
                            self.curr_attn_map_sum = curr_attn_map
                        else:
                            self.curr_attn_map_sum += curr_attn_map
                        self.curr_attn_map_cnt += 1

                if block_idx == len(self.blocks) - 1:
                    del self.prev_attn_map_sum
                    self.prev_attn_map_sum = None
                    self.prev_attn_map_cnt = 0
                    del self.curr_attn_map_sum
                    self.curr_attn_map_sum = None
                    self.curr_attn_map_cnt = 0

        x = self.head(x, t)
        x = self.unpatchify(x, (f, h, w))
        return x

    @staticmethod
    def state_dict_converter():
        return WanModelStateDictConverter()

    @classmethod
    def from_pretrained(cls, model_path, device="cpu", torch_dtype=torch.bfloat16, **kwargs):
        import os
        import re
        from safetensors.torch import load_file

        pattern = re.compile(r"diffusion_pytorch_model-\d{5}-of-\d{5}\.safetensor")
        sharded_files = [f for f in os.listdir(model_path) if pattern.match(f)]
        sharded_files = [os.path.join(model_path, f) for f in sorted(sharded_files)]
        print(f"[WanModel] Loading pretrained model from: {sharded_files}")

        state_dict = {}
        for file in sharded_files:
            state_dict.update(load_file(file))

        model_state_dict, model_kwargs = cls.state_dict_converter().from_civitai(state_dict)
        if kwargs:
            model_kwargs.update(kwargs)

        if model_kwargs.get("use_hand", False):
            hand_patch_embedding = nn.Conv3d(16, model_kwargs["dim"], kernel_size=model_kwargs["patch_size"], stride=model_kwargs["patch_size"])
            for name, param in hand_patch_embedding.named_parameters():
                model_state_dict["hand_patch_embedding." + name] = torch.zeros(param.shape)
            del hand_patch_embedding

        pe_state_dict = torch.load("./data/weights/patch_embedding.pt", weights_only=True)
        model_state_dict.update(pe_state_dict)

        with init_weights_on_device():
            model = cls(**model_kwargs)
        model.load_state_dict(model_state_dict, strict=True, assign=True)

        # model = cls(**model_kwargs)
        # missing_keys, unexpected_keys = model.load_state_dict(model_state_dict, strict=False)
        # print(f"[WanModel] => number of missing keys: {len(missing_keys)}")
        # print(f"[WanModel] => missing keys: {missing_keys}")
        # print(f"[WanModel] => number of unexpected keys: {len(unexpected_keys)}")
        # print(f"[WanModel] => unexpected keys: {unexpected_keys}")

        model.to(device=device, dtype=torch_dtype)
        return model, model_kwargs


class WanModelStateDictConverter:
    def __init__(self):
        pass

    def from_diffusers(self, state_dict):
        rename_dict = {
            "blocks.0.attn1.norm_k.weight": "blocks.0.self_attn.norm_k.weight",
            "blocks.0.attn1.norm_q.weight": "blocks.0.self_attn.norm_q.weight",
            "blocks.0.attn1.to_k.bias": "blocks.0.self_attn.k.bias",
            "blocks.0.attn1.to_k.weight": "blocks.0.self_attn.k.weight",
            "blocks.0.attn1.to_out.0.bias": "blocks.0.self_attn.o.bias",
            "blocks.0.attn1.to_out.0.weight": "blocks.0.self_attn.o.weight",
            "blocks.0.attn1.to_q.bias": "blocks.0.self_attn.q.bias",
            "blocks.0.attn1.to_q.weight": "blocks.0.self_attn.q.weight",
            "blocks.0.attn1.to_v.bias": "blocks.0.self_attn.v.bias",
            "blocks.0.attn1.to_v.weight": "blocks.0.self_attn.v.weight",
            "blocks.0.attn2.norm_k.weight": "blocks.0.cross_attn.norm_k.weight",
            "blocks.0.attn2.norm_q.weight": "blocks.0.cross_attn.norm_q.weight",
            "blocks.0.attn2.to_k.bias": "blocks.0.cross_attn.k.bias",
            "blocks.0.attn2.to_k.weight": "blocks.0.cross_attn.k.weight",
            "blocks.0.attn2.to_out.0.bias": "blocks.0.cross_attn.o.bias",
            "blocks.0.attn2.to_out.0.weight": "blocks.0.cross_attn.o.weight",
            "blocks.0.attn2.to_q.bias": "blocks.0.cross_attn.q.bias",
            "blocks.0.attn2.to_q.weight": "blocks.0.cross_attn.q.weight",
            "blocks.0.attn2.to_v.bias": "blocks.0.cross_attn.v.bias",
            "blocks.0.attn2.to_v.weight": "blocks.0.cross_attn.v.weight",
            "blocks.0.ffn.net.0.proj.bias": "blocks.0.ffn.0.bias",
            "blocks.0.ffn.net.0.proj.weight": "blocks.0.ffn.0.weight",
            "blocks.0.ffn.net.2.bias": "blocks.0.ffn.2.bias",
            "blocks.0.ffn.net.2.weight": "blocks.0.ffn.2.weight",
            "blocks.0.norm2.bias": "blocks.0.norm3.bias",
            "blocks.0.norm2.weight": "blocks.0.norm3.weight",
            "blocks.0.scale_shift_table": "blocks.0.modulation",
            "condition_embedder.text_embedder.linear_1.bias": "text_embedding.0.bias",
            "condition_embedder.text_embedder.linear_1.weight": "text_embedding.0.weight",
            "condition_embedder.text_embedder.linear_2.bias": "text_embedding.2.bias",
            "condition_embedder.text_embedder.linear_2.weight": "text_embedding.2.weight",
            "condition_embedder.time_embedder.linear_1.bias": "time_embedding.0.bias",
            "condition_embedder.time_embedder.linear_1.weight": "time_embedding.0.weight",
            "condition_embedder.time_embedder.linear_2.bias": "time_embedding.2.bias",
            "condition_embedder.time_embedder.linear_2.weight": "time_embedding.2.weight",
            "condition_embedder.time_proj.bias": "time_projection.1.bias",
            "condition_embedder.time_proj.weight": "time_projection.1.weight",
            "patch_embedding.bias": "patch_embedding.bias",
            "patch_embedding.weight": "patch_embedding.weight",
            "scale_shift_table": "head.modulation",
            "proj_out.bias": "head.head.bias",
            "proj_out.weight": "head.head.weight",
        }
        state_dict_ = {}
        for name, param in state_dict.items():
            if name in rename_dict:
                state_dict_[rename_dict[name]] = param
            else:
                name_ = ".".join(name.split(".")[:1] + ["0"] + name.split(".")[2:])
                if name_ in rename_dict:
                    name_ = rename_dict[name_]
                    name_ = ".".join(name_.split(".")[:1] + [name.split(".")[1]] + name_.split(".")[2:])
                    state_dict_[name_] = param
        if hash_state_dict_keys(state_dict) == "cb104773c6c2cb6df4f9529ad5c60d0b":
            config = {
                "model_type": "t2v",
                "patch_size": (1, 2, 2),
                "text_len": 512,
                "in_dim": 16,
                "dim": 5120,
                "ffn_dim": 13824,
                "freq_dim": 256,
                "text_dim": 4096,
                "out_dim": 16,
                "num_heads": 40,
                "num_layers": 40,
                "window_size": (-1, -1),
                "qk_norm": True,
                "cross_attn_norm": True,
                "eps": 1e-6,
            }
        else:
            config = {}
        return state_dict_, config

    def from_civitai(self, state_dict):
        if hash_state_dict_keys(state_dict) == "9269f8db9040a9d860eaca435be61814":
            config = {
                "has_image_input": False,
                "patch_size": [1, 2, 2],
                "in_dim": 16,
                "dim": 1536,
                "ffn_dim": 8960,
                "freq_dim": 256,
                "text_dim": 4096,
                "out_dim": 16,
                "num_heads": 12,
                "num_layers": 30,
                "eps": 1e-6,
            }
        elif hash_state_dict_keys(state_dict) == "aafcfd9672c3a2456dc46e1cb6e52c70":
            config = {
                "has_image_input": False,
                "patch_size": [1, 2, 2],
                "in_dim": 16,
                "dim": 5120,
                "ffn_dim": 13824,
                "freq_dim": 256,
                "text_dim": 4096,
                "out_dim": 16,
                "num_heads": 40,
                "num_layers": 40,
                "eps": 1e-6,
            }
        elif hash_state_dict_keys(state_dict) == "6bfcfb3b342cb286ce886889d519a77e":
            config = {
                "has_image_input": True,
                "patch_size": [1, 2, 2],
                "in_dim": 36,
                "dim": 5120,
                "ffn_dim": 13824,
                "freq_dim": 256,
                "text_dim": 4096,
                "out_dim": 16,
                "num_heads": 40,
                "num_layers": 40,
                "eps": 1e-6,
            }
        else:
            config = {}
        return state_dict, config
