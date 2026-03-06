# Copyright (c) Phigent Robotics. All rights reserved.
import inspect

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdet.models import DETECTORS
from uniception.models.info_sharing.base import MultiViewTransformerInput
from uniception.models.info_sharing.cross_attention_transformer import \
    MultiViewCrossAttentionTransformer

from .bevdet import BEVDet4D


def _ensure_sdpa_scale_compat():
    """Keep fused attention on while supporting torch sdpa without `scale`."""
    sdpa = F.scaled_dot_product_attention
    try:
        has_scale = 'scale' in inspect.signature(sdpa).parameters
    except (TypeError, ValueError):
        has_scale = False

    if has_scale or getattr(sdpa, '_uniception_scale_compat', False):
        return

    def _sdpa_scale_compat(query,
                           key,
                           value,
                           attn_mask=None,
                           dropout_p=0.0,
                           is_causal=False,
                           scale=None):
        if scale is not None:
            query = query * scale
        return sdpa(query,
                    key,
                    value,
                    attn_mask=attn_mask,
                    dropout_p=dropout_p,
                    is_causal=is_causal)

    _sdpa_scale_compat._uniception_scale_compat = True
    F.scaled_dot_product_attention = _sdpa_scale_compat


@DETECTORS.register_module()
class BEVDet4DMapAnythingCrossAttn(BEVDet4D):
    """BEVDet4D with map-anything multi-view cross-attention before LSS."""

    def __init__(self, mapanything_cross_attn, **kwargs):
        super(BEVDet4DMapAnythingCrossAttn, self).__init__(**kwargs)
        _ensure_sdpa_scale_compat()
        mapanything_cross_attn = dict(mapanything_cross_attn)
        # Do not load any map-anything/uniception pretrained checkpoint.
        mapanything_cross_attn['pretrained_checkpoint_path'] = None
        force_random_init = mapanything_cross_attn.pop('force_random_init', True)
        self.mapanything_cross_attn = MultiViewCrossAttentionTransformer(
            **mapanything_cross_attn)
        if force_random_init:
            self._init_cross_attn_random()

    def _init_cross_attn_random(self):
        """Explicit random init for cross-attention branch."""
        for module in self.mapanything_cross_attn.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.LayerNorm):
                nn.init.constant_(module.bias, 0)
                nn.init.constant_(module.weight, 1.0)

    def apply_mapanything_cross_attn(self, x):
        """Apply map-anything cross-attention on per-camera image features."""
        _, num_views, _, _, _ = x.shape
        features = [x[:, view_id].contiguous() for view_id in range(num_views)]
        output = self.mapanything_cross_attn(
            MultiViewTransformerInput(features=features))
        return torch.stack(output.features, dim=1)

    def prepare_bev_feat(self, img, rot, tran, intrin, post_rot, post_tran,
                         bda, mlp_input):
        x, _ = self.image_encoder(img)
        x = self.apply_mapanything_cross_attn(x)
        bev_feat, depth = self.img_view_transformer(
            [x, rot, tran, intrin, post_rot, post_tran, bda, mlp_input])
        if self.pre_process:
            bev_feat = self.pre_process_net(bev_feat)[0]
        return bev_feat, depth
