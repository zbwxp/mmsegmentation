_base_ = [
    '../_base_/models/deeppad_r50.py',
    '../_base_/datasets/pascal_context.py', '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_80k.py'
]
model = dict(
    pretrained='open-mmlab://resnet101_v1c',
    backbone=dict(depth=101),
    decode_head=dict(
        type='DeepPadHead512',
        num_classes=60,
        dyn_branch_ch=8,
        mask_head_ch=8,
        pad_out_channel_factor=512/48,
    ),
    auxiliary_head=dict(num_classes=60),
    # test_cfg=dict(mode='slide', crop_size=(480, 480), stride=(320, 320))
)
optimizer = dict(type='SGD', lr=0.004, momentum=0.9, weight_decay=0.0001)
