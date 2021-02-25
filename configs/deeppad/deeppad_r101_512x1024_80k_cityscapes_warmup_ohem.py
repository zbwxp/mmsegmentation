_base_ = [
    '../_base_/models/deeppad_r50.py',
    '../_base_/datasets/cityscapes.py', '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_80k_warmup.py'
]
model = dict(
    pretrained='open-mmlab://resnet101_v1c',
    backbone=dict(depth=101),
    decode_head=dict(
        dyn_branch_ch=8,
        mask_head_ch=8,
        # pad_out_channel_factor=2,
        sampler=dict(type='OHEMPixelSampler', thresh=0.7, min_kept=100000)
    ),
)
