_base_ = [
    '../_base_/models/deeppad_r50.py',
    '../_base_/datasets/cityscapes_512x512.py', '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_40k_warmup.py'
]
norm_cfg = dict(type='BN', requires_grad=True)
model = dict(
    backbone=dict(norm_cfg=norm_cfg,),
    decode_head=dict(
        type='DeepPadHead512',
        dyn_branch_ch=8,
        mask_head_ch=8,
        pad_out_channel_factor=512/48,
        norm_cfg=norm_cfg,
        sampler=dict(type='OHEMPixelSampler', thresh=0.7, min_kept=100000)
    ),
    auxiliary_head=dict(norm_cfg=norm_cfg)
)
