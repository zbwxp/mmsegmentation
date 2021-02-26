_base_ = [
    '../_base_/models/deeppad_r50.py',
    '../_base_/datasets/pascal_voc12_aug.py', '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_40k.py'
]
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
