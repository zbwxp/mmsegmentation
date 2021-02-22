_base_ = './deeppad_r50_512x1024_80k_cityscapes.py'
model = dict(
    pretrained='open-mmlab://resnet101_v1c',
    backbone=dict(depth=101),
    decode_head=dict(
        dyn_branch_ch=16,
        mask_head_ch=16,
        pad_out_channel_factor=2,
    ),
)
