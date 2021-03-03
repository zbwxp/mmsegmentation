_base_ = [
    '../_base_/models/deeppad_r50.py',
    '../_base_/datasets/cityscapes_512x512.py', '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_40k.py'
]
model = dict(
    pretrained='open-mmlab://resnet18_v1c',
    backbone=dict(depth=18),
    decode_head=dict(
        type='LastHopeHead',
        upsample_factor=8,
        dyn_branch_ch=8,
        mask_head_ch=8,
        c1_in_channels=64,
        c1_channels=12,
        in_channels=512,
        channels=128,
    ),
    auxiliary_head=dict(in_channels=256, channels=64)
)
