_base_ = './deeppad_r50_512x512_80k_ade20k.py'
model = dict(
    pretrained='open-mmlab://resnet101_v1c',
    backbone=dict(depth=101),
    decode_head=dict(
            type='DeepPadHead512',
            dyn_branch_ch=8,
            mask_head_ch=8,
            pad_out_channel_factor=512/48,
    ),
)
