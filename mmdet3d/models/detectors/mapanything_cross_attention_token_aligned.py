from copy import deepcopy
from functools import partial
from typing import Callable, List, Optional, Tuple, Type, Union

import torch
import torch.nn as nn

from uniception.models.info_sharing.base import (
    MultiViewTransformerInput,
    MultiViewTransformerOutput,
    UniCeptionInfoSharingBase,
)
from uniception.models.utils.intermediate_feature_return import (
    IntermediateFeatureReturner,
    feature_take_indices,
)
from uniception.models.utils.positional_encoding import PositionGetter
from uniception.models.utils.transformer_blocks import CrossAttentionBlock, Mlp


class MultiViewCrossAttentionTransformerTokenAligned(UniCeptionInfoSharingBase):
    """Token-aware cross-attention transformer compatible with MultiViewTransformerInput."""

    def __init__(
        self,
        name: str,
        input_embed_dim: int,
        num_views: int,
        size: Optional[str] = None,
        depth: int = 12,
        dim: int = 768,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        qk_norm: bool = False,
        proj_drop: float = 0.0,
        attn_drop: float = 0.0,
        init_values: Optional[float] = None,
        drop_path: float = 0.0,
        act_layer: Type[nn.Module] = nn.GELU,
        norm_layer: Union[Type[nn.Module], Callable[..., nn.Module]] = partial(
            nn.LayerNorm, eps=1e-6
        ),
        mlp_layer: Type[nn.Module] = Mlp,
        custom_positional_encoding: Optional[Callable] = None,
        norm_cross_tokens: bool = True,
        use_scalable_softmax: bool = False,
        use_entropy_scaling: bool = False,
        base_token_count_for_entropy_scaling: int = 444,
        entropy_scaling_growth_factor: float = 1.4,
        pretrained_checkpoint_path: Optional[str] = None,
        gradient_checkpointing: bool = False,
        *args,
        **kwargs,
    ):
        super().__init__(name=name, size=size, *args, **kwargs)
        self.input_embed_dim = input_embed_dim
        self.num_views = num_views
        self.depth = depth
        self.dim = dim
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.qkv_bias = qkv_bias
        self.qk_norm = qk_norm
        self.proj_drop = proj_drop
        self.attn_drop = attn_drop
        self.init_values = init_values
        self.drop_path = drop_path
        self.act_layer = act_layer
        self.norm_layer = norm_layer
        self.mlp_layer = mlp_layer
        self.custom_positional_encoding = custom_positional_encoding
        self.norm_cross_tokens = norm_cross_tokens
        self.use_scalable_softmax = use_scalable_softmax
        self.use_entropy_scaling = use_entropy_scaling
        self.base_token_count_for_entropy_scaling = base_token_count_for_entropy_scaling
        self.entropy_scaling_growth_factor = entropy_scaling_growth_factor
        self.pretrained_checkpoint_path = pretrained_checkpoint_path
        self.gradient_checkpointing = gradient_checkpointing

        if self.input_embed_dim != self.dim:
            self.proj_embed = nn.Linear(self.input_embed_dim, self.dim, bias=True)
        else:
            self.proj_embed = nn.Identity()

        cross_attention_blocks = nn.ModuleList(
            [
                CrossAttentionBlock(
                    dim=self.dim,
                    num_heads=self.num_heads,
                    mlp_ratio=self.mlp_ratio,
                    qkv_bias=self.qkv_bias,
                    qk_norm=self.qk_norm,
                    proj_drop=self.proj_drop,
                    attn_drop=self.attn_drop,
                    init_values=self.init_values,
                    drop_path=self.drop_path,
                    act_layer=self.act_layer,
                    norm_layer=self.norm_layer,
                    mlp_layer=self.mlp_layer,
                    custom_positional_encoding=self.custom_positional_encoding,
                    norm_cross_tokens=self.norm_cross_tokens,
                    use_scalable_softmax=self.use_scalable_softmax,
                    use_entropy_scaling=self.use_entropy_scaling,
                    base_token_count_for_entropy_scaling=self.base_token_count_for_entropy_scaling,
                    entropy_scaling_growth_factor=self.entropy_scaling_growth_factor,
                )
                for _ in range(self.depth)
            ]
        )
        self.multi_view_branches = nn.ModuleList([cross_attention_blocks])
        for _ in range(1, self.num_views):
            self.multi_view_branches.append(deepcopy(cross_attention_blocks))

        self.norm = self.norm_layer(self.dim)
        if self.custom_positional_encoding is not None:
            self.position_getter = PositionGetter()

        self.initialize_weights()
        if self.gradient_checkpointing:
            for view_idx in range(self.num_views):
                for block_idx in range(self.depth):
                    self.multi_view_branches[view_idx][block_idx] = (
                        self.wrap_module_with_gradient_checkpointing(
                            self.multi_view_branches[view_idx][block_idx]
                        )
                    )

        if self.pretrained_checkpoint_path is not None:
            ckpt = torch.load(self.pretrained_checkpoint_path, weights_only=False)
            self.load_state_dict(ckpt["model"])

    def initialize_weights(self):
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def _flatten_input(
        self, model_input: MultiViewTransformerInput
    ) -> Tuple[List[torch.Tensor], int, int, int, int, int, int]:
        assert len(model_input.features) == self.num_views, (
            f"Expected {self.num_views} views, got {len(model_input.features)}"
        )
        assert all(
            feat.shape[1] == self.input_embed_dim for feat in model_input.features
        ), f"All views must have input dimension {self.input_embed_dim}"
        assert all(feat.ndim == 4 for feat in model_input.features), (
            "All views must have 4 dimensions (N, C, H, W)"
        )

        b, _, h, w = model_input.features[0].shape
        spatial_tokens_per_view = h * w

        per_view_tokens = model_input.additional_input_tokens_per_view
        num_per_view_tokens = 0
        if per_view_tokens is not None:
            assert len(per_view_tokens) == self.num_views, (
                "additional_input_tokens_per_view length must equal num_views"
            )
            assert all(tokens.ndim == 3 for tokens in per_view_tokens), (
                "Per-view additional tokens must be (B, C, T)"
            )
            assert all(tokens.shape[0] == b for tokens in per_view_tokens), (
                "Batch size mismatch for additional_input_tokens_per_view"
            )
            assert all(tokens.shape[1] == self.input_embed_dim for tokens in per_view_tokens), (
                f"Per-view additional tokens must have dim {self.input_embed_dim}"
            )
            num_per_view_tokens = per_view_tokens[0].shape[2]
            assert all(tokens.shape[2] == num_per_view_tokens for tokens in per_view_tokens), (
                "Per-view token count must match across views"
            )

        global_tokens = model_input.additional_input_tokens
        num_global_tokens = 0
        if global_tokens is not None:
            assert global_tokens.ndim == 3, (
                "additional_input_tokens must be (B, C, T)"
            )
            assert global_tokens.shape[0] == b, "Batch size mismatch for additional_input_tokens"
            assert global_tokens.shape[1] == self.input_embed_dim, (
                f"additional_input_tokens must have dim {self.input_embed_dim}"
            )
            num_global_tokens = global_tokens.shape[2]
            global_tokens = global_tokens.permute(0, 2, 1).contiguous()

        if self.custom_positional_encoding is not None and (
            num_per_view_tokens > 0 or num_global_tokens > 0
        ):
            raise ValueError(
                "custom_positional_encoding with additional tokens is not supported "
                "in this cross-attention implementation."
            )

        view_sequences = []
        for view_idx, view_feat in enumerate(model_input.features):
            seq = view_feat.permute(0, 2, 3, 1).reshape(
                b, spatial_tokens_per_view, self.input_embed_dim
            ).contiguous()
            if num_per_view_tokens > 0:
                per_view_seq = per_view_tokens[view_idx].permute(0, 2, 1).contiguous()
                seq = torch.cat([seq, per_view_seq], dim=1)
            if num_global_tokens > 0:
                seq = torch.cat([seq, global_tokens], dim=1)
            view_sequences.append(seq)

        return (
            view_sequences,
            b,
            h,
            w,
            spatial_tokens_per_view,
            num_per_view_tokens,
            num_global_tokens,
        )

    def _build_positions(
        self, b: int, h: int, w: int, device: torch.device
    ) -> List[Optional[torch.Tensor]]:
        if self.custom_positional_encoding is None:
            return [None] * self.num_views
        return [self.position_getter(b, h, w, device) for _ in range(self.num_views)]

    def _sync_global_tokens(
        self,
        view_sequences: List[torch.Tensor],
        view_token_count_without_global: int,
        num_global_tokens: int,
    ) -> List[torch.Tensor]:
        if num_global_tokens == 0:
            return view_sequences
        global_chunks = [
            seq[:, view_token_count_without_global : view_token_count_without_global + num_global_tokens, :]
            for seq in view_sequences
        ]
        shared_global = torch.stack(global_chunks, dim=0).mean(dim=0)
        synced = []
        for seq in view_sequences:
            updated = seq.clone()
            updated[
                :, view_token_count_without_global : view_token_count_without_global + num_global_tokens, :
            ] = shared_global
            synced.append(updated)
        return synced

    def _pack_output(
        self,
        view_sequences: List[torch.Tensor],
        b: int,
        h: int,
        w: int,
        spatial_tokens_per_view: int,
        num_per_view_tokens: int,
        num_global_tokens: int,
    ) -> MultiViewTransformerOutput:
        out_features = []
        out_per_view_tokens = [] if num_per_view_tokens > 0 else None
        per_view_end = spatial_tokens_per_view + num_per_view_tokens

        for seq in view_sequences:
            spatial_seq = seq[:, :spatial_tokens_per_view, :]
            feat = spatial_seq.reshape(b, h, w, self.dim).permute(0, 3, 1, 2).contiguous()
            out_features.append(feat)
            if num_per_view_tokens > 0:
                view_tokens = seq[:, spatial_tokens_per_view:per_view_end, :].permute(
                    0, 2, 1
                ).contiguous()
                out_per_view_tokens.append(view_tokens)

        out_global_tokens = None
        if num_global_tokens > 0:
            global_chunks = [
                seq[:, per_view_end : per_view_end + num_global_tokens, :]
                for seq in view_sequences
            ]
            out_global_tokens = torch.stack(global_chunks, dim=0).mean(dim=0)
            out_global_tokens = out_global_tokens.permute(0, 2, 1).contiguous()

        return MultiViewTransformerOutput(
            features=out_features,
            additional_token_features=out_global_tokens,
            additional_token_features_per_view=out_per_view_tokens,
        )

    def _forward_core(
        self,
        model_input: MultiViewTransformerInput,
        take_indices: Optional[List[int]] = None,
        norm_intermediate: bool = True,
    ) -> Tuple[MultiViewTransformerOutput, List[MultiViewTransformerOutput]]:
        (
            view_sequences,
            b,
            h,
            w,
            spatial_tokens_per_view,
            num_per_view_tokens,
            num_global_tokens,
        ) = self._flatten_input(model_input)
        view_sequences = [self.proj_embed(seq) for seq in view_sequences]

        positions = self._build_positions(
            b, h, w, model_input.features[0].device
        )
        per_view_end = spatial_tokens_per_view + num_per_view_tokens
        intermediate_outputs = []

        for depth_idx in range(self.depth):
            updated = []
            for view_idx, curr_seq in enumerate(view_sequences):
                other = [view_sequences[i] for i in range(self.num_views) if i != view_idx]
                other_seq = torch.cat(other, dim=1)
                curr_pos = positions[view_idx]
                other_pos = (
                    torch.cat([positions[i] for i in range(self.num_views) if i != view_idx], dim=1)
                    if curr_pos is not None
                    else None
                )
                out_seq = self.multi_view_branches[view_idx][depth_idx](
                    curr_seq, other_seq, curr_pos, other_pos
                )
                updated.append(out_seq)

            view_sequences = self._sync_global_tokens(
                updated, per_view_end, num_global_tokens
            )

            if take_indices is not None and depth_idx in take_indices:
                curr_for_out = (
                    [self.norm(seq) for seq in view_sequences]
                    if norm_intermediate
                    else [seq.clone() for seq in view_sequences]
                )
                intermediate_outputs.append(
                    self._pack_output(
                        curr_for_out,
                        b,
                        h,
                        w,
                        spatial_tokens_per_view,
                        num_per_view_tokens,
                        num_global_tokens,
                    )
                )

        final_normed = [self.norm(seq) for seq in view_sequences]
        final_output = self._pack_output(
            final_normed,
            b,
            h,
            w,
            spatial_tokens_per_view,
            num_per_view_tokens,
            num_global_tokens,
        )
        return final_output, intermediate_outputs

    def forward(
        self,
        model_input: MultiViewTransformerInput,
    ) -> MultiViewTransformerOutput:
        final_output, _ = self._forward_core(model_input)
        return final_output


class MultiViewCrossAttentionTransformerIFRTokenAligned(
    MultiViewCrossAttentionTransformerTokenAligned, IntermediateFeatureReturner
):
    """IFR variant with token-aware I/O."""

    def __init__(
        self,
        name: str,
        input_embed_dim: int,
        num_views: int,
        size: Optional[str] = None,
        depth: int = 12,
        dim: int = 768,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        qk_norm: bool = False,
        proj_drop: float = 0.0,
        attn_drop: float = 0.0,
        init_values: Optional[float] = None,
        drop_path: float = 0.0,
        act_layer: nn.Module = nn.GELU,
        norm_layer: nn.Module = partial(nn.LayerNorm, eps=1e-6),
        mlp_layer: nn.Module = Mlp,
        custom_positional_encoding: Callable = None,
        norm_cross_tokens: bool = True,
        use_scalable_softmax: bool = False,
        use_entropy_scaling: bool = False,
        base_token_count_for_entropy_scaling: int = 444,
        entropy_scaling_growth_factor: float = 1.4,
        pretrained_checkpoint_path: str = None,
        indices: Optional[Union[int, List[int]]] = None,
        norm_intermediate: bool = True,
        intermediates_only: bool = False,
        gradient_checkpointing: bool = False,
        *args,
        **kwargs,
    ):
        MultiViewCrossAttentionTransformerTokenAligned.__init__(
            self,
            name=name,
            input_embed_dim=input_embed_dim,
            num_views=num_views,
            size=size,
            depth=depth,
            dim=dim,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            proj_drop=proj_drop,
            attn_drop=attn_drop,
            init_values=init_values,
            drop_path=drop_path,
            act_layer=act_layer,
            norm_layer=norm_layer,
            mlp_layer=mlp_layer,
            custom_positional_encoding=custom_positional_encoding,
            norm_cross_tokens=norm_cross_tokens,
            use_scalable_softmax=use_scalable_softmax,
            use_entropy_scaling=use_entropy_scaling,
            base_token_count_for_entropy_scaling=base_token_count_for_entropy_scaling,
            entropy_scaling_growth_factor=entropy_scaling_growth_factor,
            pretrained_checkpoint_path=pretrained_checkpoint_path,
            gradient_checkpointing=gradient_checkpointing,
            *args,
            **kwargs,
        )
        IntermediateFeatureReturner.__init__(
            self,
            indices=indices,
            norm_intermediate=norm_intermediate,
            intermediates_only=intermediates_only,
        )

    def forward(
        self,
        model_input: MultiViewTransformerInput,
    ) -> Union[
        List[MultiViewTransformerOutput],
        Tuple[MultiViewTransformerOutput, List[MultiViewTransformerOutput]],
    ]:
        take_indices, _ = feature_take_indices(self.depth, self.indices)
        final_output, intermediate_outputs = self._forward_core(
            model_input,
            take_indices=take_indices,
            norm_intermediate=self.norm_intermediate,
        )
        if self.intermediates_only:
            return intermediate_outputs
        return final_output, intermediate_outputs
