_base_ = [
    '../_base_/models/deeppad_r50.py',
    '../_base_/datasets/cityscapes.py', '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_80k.py'
]
model = dict(
    pretrained='open-mmlab://resnet101_v1c',
    backbone=dict(depth=101),
    decode_head=dict(
        type='BilinearHead',
        pad_out_channel_factor=512/48,
    ),
)
