_base_ = ['./bevdet-r50-mapanything-cross-attn-3frame.py']

model = dict(
    mapanything_cross_attn=dict(
        depth=12,
    ),
)
