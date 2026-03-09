# Copyright (c) Phigent Robotics. All rights reserved.
import torch
import torch.nn as nn
from mmdet.models import DETECTORS

from .bevdet import BEVDet


class _CameraCrossAttentionBlock(nn.Module):

    def __init__(self,
                 embed_dims,
                 num_heads,
                 attn_drop=0.0,
                 proj_drop=0.0,
                 ffn_ratio=2.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dims)
        self.attn = nn.MultiheadAttention(
            embed_dim=embed_dims,
            num_heads=num_heads,
            dropout=attn_drop,
            batch_first=True)
        self.proj_drop = nn.Dropout(proj_drop)
        self.norm2 = nn.LayerNorm(embed_dims)
        hidden_dim = int(embed_dims * ffn_ratio)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dims, hidden_dim),
            nn.GELU(),
            nn.Dropout(proj_drop),
            nn.Linear(hidden_dim, embed_dims),
            nn.Dropout(proj_drop))

    def forward(self, x):
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm, need_weights=False)
        x = x + self.proj_drop(attn_out)
        x = x + self.ffn(self.norm2(x))
        return x


class PreLSSCameraCrossAttention(nn.Module):
    """Cross-attend camera-view features at each spatial location."""

    def __init__(self,
                 embed_dims,
                 num_heads,
                 num_layers=1,
                 attn_drop=0.0,
                 proj_drop=0.0,
                 ffn_ratio=2.0,
                 max_cams=12,
                 use_camera_embedding=True):
        super().__init__()
        if embed_dims % num_heads != 0:
            raise ValueError('embed_dims must be divisible by num_heads, '
                             f'got {embed_dims} and {num_heads}.')
        self.embed_dims = embed_dims
        self.max_cams = max_cams
        self.use_camera_embedding = use_camera_embedding
        if self.use_camera_embedding:
            self.camera_embed = nn.Parameter(torch.zeros(max_cams, embed_dims))
            nn.init.trunc_normal_(self.camera_embed, std=0.02)
        self.layers = nn.ModuleList([
            _CameraCrossAttentionBlock(
                embed_dims=embed_dims,
                num_heads=num_heads,
                attn_drop=attn_drop,
                proj_drop=proj_drop,
                ffn_ratio=ffn_ratio) for _ in range(num_layers)
        ])

    def forward(self, x):
        # x: [B, Ncam, C, H, W]
        if x.dim() != 5:
            raise ValueError(f'Expect x with shape [B, N, C, H, W], got {x.shape}.')
        b, n, c, h, w = x.shape
        if c != self.embed_dims:
            raise ValueError('Input channel mismatch: '
                             f'expect {self.embed_dims}, got {c}.')
        if self.use_camera_embedding and n > self.max_cams:
            raise ValueError('Number of cameras exceeds max_cams: '
                             f'got {n}, max {self.max_cams}.')

        # Each spatial location (h, w) forms a camera-token sequence.
        tokens = x.permute(0, 3, 4, 1, 2).reshape(b * h * w, n, c)
        if self.use_camera_embedding:
            camera_embed = self.camera_embed[:n].to(dtype=tokens.dtype)
            tokens = tokens + camera_embed.unsqueeze(0)
        for layer in self.layers:
            tokens = layer(tokens)
        x = tokens.view(b, h, w, n, c).permute(0, 3, 4, 1, 2).contiguous()
        return x


@DETECTORS.register_module()
class BEVDetPreLSSCrossAttn(BEVDet):
    """Apply cross-attention to image features before feeding LSS."""

    def __init__(self, pre_lss_cross_attn, **kwargs):
        super(BEVDetPreLSSCrossAttn, self).__init__(**kwargs)
        pre_lss_cross_attn = dict(pre_lss_cross_attn)
        self.pre_lss_cross_attn = PreLSSCameraCrossAttention(
            **pre_lss_cross_attn)

    def extract_img_feat(self, img, img_metas, **kwargs):
        """Extract features of images with pre-LSS cross-attention."""
        img = self.prepare_inputs(img)
        x, _ = self.image_encoder(img[0])
        x = self.pre_lss_cross_attn(x)
        x, depth = self.img_view_transformer([x] + img[1:7])
        x = self.bev_encoder(x)
        return [x], depth
