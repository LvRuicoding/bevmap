custom_imports = dict(
    imports=['mmdet3d.models.detectors.bevdet_pre_lss_cross_attn'],
    allow_failed_imports=False)

_base_ = ['./bevdet-r50.py']

model = dict(
    type='BEVDetPreLSSCrossAttn',
    pre_lss_cross_attn=dict(
        embed_dims=256,
        num_heads=8,
        num_layers=1,
        attn_drop=0.0,
        proj_drop=0.0,
        ffn_ratio=2.0,
        max_cams=6,
        use_camera_embedding=True))
