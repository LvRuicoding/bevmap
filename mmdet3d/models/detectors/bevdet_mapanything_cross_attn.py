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

    def apply_mapanything_cross_attn(self, img_feats):
        """Apply cross-attention over all frames/views, then residual fuse."""
        num_frame = len(img_feats)
        b, n, c, h, w = img_feats[0].shape
        feat_all = torch.cat(img_feats, dim=1)
        total_views = feat_all.shape[1]
        if total_views != self.mapanything_cross_attn.num_views:
            raise ValueError(
                f'Configured num_views={self.mapanything_cross_attn.num_views}, '
                f'but got {total_views} views ({num_frame} frames x {n} cams).')
        features = [feat_all[:, view_id].contiguous()
                    for view_id in range(total_views)]
        output = self.mapanything_cross_attn(
            MultiViewTransformerInput(features=features))
        attn_all = torch.stack(output.features, dim=1)
        feat_all = feat_all + attn_all
        feat_all = feat_all.view(b, num_frame, n, c, h, w)
        return [feat.contiguous() for feat in feat_all.unbind(dim=1)]

    def forward_lss_from_feature(self, x, rot, tran, intrin, post_rot,
                                 post_tran, bda, mlp_input):
        x = x.contiguous()
        bev_feat, depth = self.img_view_transformer(
            [x, rot, tran, intrin, post_rot, post_tran, bda, mlp_input])
        if self.pre_process:
            bev_feat = self.pre_process_net(bev_feat)[0]
        return bev_feat, depth

    def extract_img_feat(self,
                         img,
                         img_metas,
                         pred_prev=False,
                         sequential=False,
                         **kwargs):
        if sequential or pred_prev:
            return super(BEVDet4DMapAnythingCrossAttn, self).extract_img_feat(
                img, img_metas, pred_prev=pred_prev, sequential=sequential,
                **kwargs)

        imgs, sensor2keyegos, ego2globals, intrins, post_rots, post_trans, \
        bda, _ = self.prepare_inputs(img)

        img_feat_list = []
        key_frame = True
        for img_frame in imgs:
            if key_frame or self.with_prev:
                if key_frame:
                    x, _ = self.image_encoder(img_frame)
                else:
                    with torch.no_grad():
                        x, _ = self.image_encoder(img_frame)
            else:
                x = torch.zeros_like(img_feat_list[0])
            img_feat_list.append(x)
            key_frame = False

        img_feat_list = self.apply_mapanything_cross_attn(img_feat_list)

        bev_feat_list = []
        depth_list = []
        key_frame = True
        for x, sensor2keyego, ego2global, intrin, post_rot, post_tran in zip(
                img_feat_list, sensor2keyegos, ego2globals, intrins,
                post_rots, post_trans):
            if key_frame or self.with_prev:
                if self.align_after_view_transfromation:
                    sensor2keyego, ego2global = sensor2keyegos[0], ego2globals[0]
                mlp_input = self.img_view_transformer.get_mlp_input(
                    sensor2keyegos[0], ego2globals[0], intrin, post_rot,
                    post_tran, bda)
                inputs_curr = (x, sensor2keyego, ego2global, intrin, post_rot,
                               post_tran, bda, mlp_input)
                if key_frame:
                    bev_feat, depth = self.forward_lss_from_feature(*inputs_curr)
                else:
                    with torch.no_grad():
                        bev_feat, depth = self.forward_lss_from_feature(
                            *inputs_curr)
            else:
                bev_feat = torch.zeros_like(bev_feat_list[0])
                depth = None
            bev_feat_list.append(bev_feat)
            depth_list.append(depth)
            key_frame = False

        if self.align_after_view_transfromation:
            for adj_id in range(1, self.num_frame):
                bev_feat_list[adj_id] = \
                    self.shift_feature(bev_feat_list[adj_id],
                                       [sensor2keyegos[0],
                                        sensor2keyegos[adj_id]],
                                       bda)
        bev_feat = torch.cat(bev_feat_list, dim=1)
        x = self.bev_encoder(bev_feat)
        return [x], depth_list[0]
