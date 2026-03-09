# Copyright (c) Phigent Robotics. All rights reserved.
import inspect

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdet.models import DETECTORS
from uniception.models.info_sharing.base import MultiViewTransformerInput

from .bevdet import BEVDet4D
from .mapanything_cross_attention_token_aligned import \
    MultiViewCrossAttentionTransformerIFRTokenAligned, \
    MultiViewCrossAttentionTransformerTokenAligned


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
    """Temporal cross-attn context, but single-frame LSS/BEV detection."""

    def __init__(self,
                 mapanything_cross_attn,
                 explicit_geo_encoding=None,
                 **kwargs):
        super(BEVDet4DMapAnythingCrossAttn, self).__init__(**kwargs)
        _ensure_sdpa_scale_compat()
        mapanything_cross_attn = dict(mapanything_cross_attn)
        # Do not load any map-anything/uniception pretrained checkpoint.
        mapanything_cross_attn['pretrained_checkpoint_path'] = None
        force_random_init = mapanything_cross_attn.pop('force_random_init', True)
        use_ifr = mapanything_cross_attn.pop('use_ifr', False)
        self.use_ifr_intermediates = mapanything_cross_attn.pop(
            'use_ifr_intermediates', True)
        self.use_token_features = mapanything_cross_attn.pop(
            'use_token_features', True)
        self.use_scale_token = mapanything_cross_attn.pop(
            'use_scale_token', self.use_token_features)
        self.use_register_tokens = mapanything_cross_attn.pop(
            'use_register_tokens', self.use_token_features)
        self.num_register_tokens = int(
            mapanything_cross_attn.pop('num_register_tokens', 4))
        self.ifr_align_fusion = mapanything_cross_attn.pop(
            'ifr_align_fusion', True)
        self.cross_attn_embed_dim = mapanything_cross_attn['input_embed_dim']
        self.ifr_intermediate_count = self._infer_ifr_intermediate_count(
            mapanything_cross_attn.get('indices', None),
            mapanything_cross_attn.get('depth', None),
            use_ifr)
        # MapAnything DPT path: len(indices)==2 keeps encoder feature as extra level.
        self.ifr_use_encoder_feature = self.ifr_intermediate_count == 2
        self.ifr_fusion_num_terms = (
            1 + self.ifr_intermediate_count + int(self.ifr_use_encoder_feature))
        geo_cfg = dict(
            enabled=True,
            include_ego2global=True,
            hidden_dim=self.cross_attn_embed_dim)
        if explicit_geo_encoding is not None:
            geo_cfg.update(explicit_geo_encoding)
        self.use_explicit_geo_encoding = geo_cfg['enabled']
        self.include_ego2global = geo_cfg['include_ego2global']
        geo_in_dim = 12 + 9 + 9 + 3 + 12
        if self.include_ego2global:
            geo_in_dim += 12
        self.geo_encoder = nn.Sequential(
            nn.LayerNorm(geo_in_dim),
            nn.Linear(geo_in_dim, geo_cfg['hidden_dim']),
            nn.ReLU(inplace=True),
            nn.Linear(geo_cfg['hidden_dim'], self.cross_attn_embed_dim))
        self.scale_token = None
        if self.use_scale_token:
            self.scale_token = nn.Parameter(torch.zeros(self.cross_attn_embed_dim))
            nn.init.trunc_normal_(self.scale_token, std=0.02)
        self.register_token_proj = None
        if self.use_register_tokens and self.num_register_tokens > 0:
            self.register_token_proj = nn.Sequential(
                nn.LayerNorm(self.cross_attn_embed_dim),
                nn.Linear(self.cross_attn_embed_dim,
                          self.cross_attn_embed_dim * self.num_register_tokens))
        self.ifr_fusion = None
        if (use_ifr and self.use_ifr_intermediates and self.ifr_align_fusion and
                self.ifr_intermediate_count > 0):
            self.ifr_fusion = nn.Conv2d(
                self.cross_attn_embed_dim * self.ifr_fusion_num_terms,
                self.cross_attn_embed_dim,
                kernel_size=1,
                bias=True)
        if use_ifr:
            self.mapanything_cross_attn = \
                MultiViewCrossAttentionTransformerIFRTokenAligned(
                    **mapanything_cross_attn)
        else:
            self.mapanything_cross_attn = \
                MultiViewCrossAttentionTransformerTokenAligned(
                    **mapanything_cross_attn)
        if force_random_init:
            self._init_cross_attn_random()

    def _infer_ifr_intermediate_count(self, indices_cfg, depth_cfg, use_ifr):
        if not use_ifr:
            return 0
        if isinstance(indices_cfg, list):
            return len(indices_cfg)
        if isinstance(indices_cfg, int):
            return max(indices_cfg, 0)
        if isinstance(depth_cfg, int):
            return max(depth_cfg, 0)
        return 0

    def _init_cross_attn_random(self):
        """Explicit random init for cross-attention branch."""
        for module in self.mapanything_cross_attn.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Conv2d):
                nn.init.kaiming_uniform_(module.weight, a=1)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.LayerNorm):
                nn.init.constant_(module.bias, 0)
                nn.init.constant_(module.weight, 1.0)
        for module in self.geo_encoder.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.LayerNorm):
                nn.init.constant_(module.bias, 0)
                nn.init.constant_(module.weight, 1.0)
        if self.register_token_proj is not None:
            for module in self.register_token_proj.modules():
                if isinstance(module, nn.Linear):
                    nn.init.xavier_uniform_(module.weight)
                    if module.bias is not None:
                        nn.init.constant_(module.bias, 0)
                elif isinstance(module, nn.LayerNorm):
                    nn.init.constant_(module.bias, 0)
                    nn.init.constant_(module.weight, 1.0)
        if self.ifr_fusion is not None:
            nn.init.kaiming_uniform_(self.ifr_fusion.weight, a=1)
            if self.ifr_fusion.bias is not None:
                nn.init.constant_(self.ifr_fusion.bias, 0)
        if self.scale_token is not None:
            nn.init.trunc_normal_(self.scale_token, std=0.02)

    def _build_geometry_vector(self, sensor2keyego, ego2global, intrin,
                               post_rot, post_tran, bda):
        b, n = sensor2keyego.shape[:2]
        parts = [sensor2keyego[:, :, :3, :].reshape(b, n, -1)]
        if self.include_ego2global:
            parts.append(ego2global[:, :, :3, :].reshape(b, n, -1))
        parts.extend([
            intrin.reshape(b, n, -1),
            post_rot[:, :, :3, :3].reshape(b, n, -1),
            post_tran.reshape(b, n, -1),
            bda[:, :3, :].reshape(b, 1, -1).expand(-1, n, -1),
        ])
        return torch.cat(parts, dim=-1)

    def _encode_geometry(self, sensor2keyegos, ego2globals, intrins,
                         post_rots, post_trans, bda, dtype):
        if not self.use_explicit_geo_encoding:
            return None
        geo_feats = []
        for sensor2keyego, ego2global, intrin, post_rot, post_tran in zip(
                sensor2keyegos, ego2globals, intrins, post_rots, post_trans):
            geo = self._build_geometry_vector(
                sensor2keyego, ego2global, intrin, post_rot, post_tran, bda)
            geo = self.geo_encoder(geo.float()).to(dtype=dtype)
            geo_feats.append(geo.unsqueeze(-1).unsqueeze(-1))
        return torch.cat(geo_feats, dim=1)

    def _build_additional_tokens(self, feat_all_attn_input):
        b, total_views, c, h, w = feat_all_attn_input.shape
        scale_token = None
        if self.scale_token is not None:
            scale_token = self.scale_token.unsqueeze(0).unsqueeze(-1).expand(
                b, -1, 1).to(
                    device=feat_all_attn_input.device,
                    dtype=feat_all_attn_input.dtype)
        register_tokens = None
        if self.register_token_proj is not None and self.num_register_tokens > 0:
            pooled = F.adaptive_avg_pool2d(
                feat_all_attn_input.reshape(b * total_views, c, h, w), 1)
            pooled = pooled.reshape(b, total_views, c)
            register_tokens = []
            for view_id in range(total_views):
                tokens = self.register_token_proj(
                    pooled[:, view_id].float()).to(
                        dtype=feat_all_attn_input.dtype)
                tokens = tokens.view(
                    b, c, self.num_register_tokens).contiguous()
                register_tokens.append(tokens)
        return scale_token, register_tokens

    def _select_intermediate_outputs(self, intermediate_outputs):
        if self.ifr_intermediate_count <= 0:
            return []
        if len(intermediate_outputs) <= self.ifr_intermediate_count:
            return intermediate_outputs
        return intermediate_outputs[-self.ifr_intermediate_count:]

    def _fuse_ifr_features(self, feat_all_attn_input, final_output,
                           intermediate_outputs):
        if (not self.use_ifr_intermediates) or len(intermediate_outputs) == 0:
            return final_output.features
        if self.ifr_fusion is None:
            fused_features = []
            for view_id in range(len(final_output.features)):
                terms = [final_output.features[view_id]]
                for inter in intermediate_outputs:
                    terms.append(inter.features[view_id])
                feat = terms[0]
                for term in terms[1:]:
                    feat = feat + term
                fused_features.append(feat / len(terms))
            return fused_features
        selected_intermediates = self._select_intermediate_outputs(
            intermediate_outputs)
        fused_features = []
        total_views = len(final_output.features)
        for view_id in range(total_views):
            terms = []
            if self.ifr_use_encoder_feature:
                terms.append(feat_all_attn_input[:, view_id])
            for inter in selected_intermediates:
                terms.append(inter.features[view_id])
            terms.append(final_output.features[view_id])
            if len(terms) != self.ifr_fusion_num_terms:
                feat = terms[0]
                for term in terms[1:]:
                    feat = feat + term
                fused_features.append(feat / len(terms))
            else:
                fused = self.ifr_fusion(torch.cat(terms, dim=1))
                fused_features.append(fused)
        return fused_features

    def apply_mapanything_cross_attn(self, img_feats, sensor2keyegos,
                                     ego2globals, intrins, post_rots,
                                     post_trans, bda):
        """Apply cross-attention over all frames/views, then residual fuse."""
        num_frame = len(img_feats)
        b, n, c, h, w = img_feats[0].shape
        feat_all_orig = torch.cat(img_feats, dim=1)
        total_views = feat_all_orig.shape[1]
        if total_views != self.mapanything_cross_attn.num_views:
            raise ValueError(
                f'Configured num_views={self.mapanything_cross_attn.num_views}, '
                f'but got {total_views} views ({num_frame} frames x {n} cams).')

        geo_all = self._encode_geometry(
            sensor2keyegos, ego2globals, intrins, post_rots, post_trans, bda,
            feat_all_orig.dtype)
        feat_all_attn_input = feat_all_orig if geo_all is None else \
            (feat_all_orig + geo_all)

        features = [feat_all_attn_input[:, view_id].contiguous()
                    for view_id in range(total_views)]
        scale_token = None
        register_tokens = None
        if self.use_token_features:
            scale_token, register_tokens = self._build_additional_tokens(
                feat_all_attn_input)
        output = self.mapanything_cross_attn(
            MultiViewTransformerInput(
                features=features,
                additional_input_tokens=scale_token,
                additional_input_tokens_per_view=register_tokens))
        if isinstance(output, tuple):
            final_output, intermediate_outputs = output
            output_features = self._fuse_ifr_features(
                feat_all_attn_input, final_output, intermediate_outputs)
        elif isinstance(output, list):
            if self.use_ifr_intermediates and len(output) > 0:
                final_output = output[-1]
                intermediate_outputs = output[:-1]
                output_features = self._fuse_ifr_features(
                    feat_all_attn_input, final_output, intermediate_outputs)
            else:
                output_features = output[-1].features
        else:
            output_features = output.features
        attn_all = torch.stack(output_features, dim=1)
        feat_all = feat_all_orig + attn_all
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

        img_feat_list = self.apply_mapanything_cross_attn(
            img_feat_list, sensor2keyegos, ego2globals, intrins,
            post_rots, post_trans, bda)
        # Only key-frame features enter LSS; history is used for cross-attn only.
        x_key = img_feat_list[0]
        sensor2keyego_key = sensor2keyegos[0]
        ego2global_key = ego2globals[0]
        intrin_key = intrins[0]
        post_rot_key = post_rots[0]
        post_tran_key = post_trans[0]
        mlp_input = self.img_view_transformer.get_mlp_input(
            sensor2keyegos[0], ego2globals[0], intrin_key, post_rot_key,
            post_tran_key, bda)
        bev_feat, depth = self.forward_lss_from_feature(
            x_key, sensor2keyego_key, ego2global_key, intrin_key, post_rot_key,
            post_tran_key, bda, mlp_input)
        x = self.bev_encoder(bev_feat)
        return [x], depth
