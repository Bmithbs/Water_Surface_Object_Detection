_base_ = './yolox_s_8x8_300e_coco.py'

pretrained = 'checkpoints/swin_tiny_patch4_window7_224.pth'
# model settings
model = dict(
    # backbone=dict(deepen_factor=1.0, widen_factor=1.0),
    random_size_range=(3, 15),
    backbone=dict( # Swin-t
        _delete_=True,
        type='SwinTransformer',
        embed_dims=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=7,
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.2,
        patch_norm=True,
        out_indices=(1, 2, 3), # 192, 384, 768
        with_cp=False,
        convert_weights=True,
        init_cfg=dict(type='Pretrained', checkpoint=pretrained)),
    neck=dict(
        in_channels=[192, 384, 768], out_channels=256, num_csp_blocks=3),
    bbox_head=dict(in_channels=256, feat_channels=256))
