_base_ = './deeppad_r50_512x1024_40k_cityscapes.py'
model = dict(
    pretrained='open-mmlab://resnet101_v1c',
    backbone=dict(depth=101),
    decode_head=dict(
        dyn_branch_ch=32,
        mask_head_ch=32,
    ),
)
