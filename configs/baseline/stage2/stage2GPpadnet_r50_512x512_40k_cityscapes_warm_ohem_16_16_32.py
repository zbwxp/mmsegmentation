_base_ = [
    '../../_base_/models/deeppad_r50.py',
    '../../_base_/datasets/cityscapes_512x512.py', '../../_base_/default_runtime.py',
    '../../_base_/schedules/schedule_40k_warmup.py'
]
norm_cfg = dict(type='BN', requires_grad=True)
model = dict(
    backbone=dict(
        norm_cfg=norm_cfg,
    ),
    decode_head=dict(
        type='Stage2PADHead_r50_GP_fast',
        upsample_factor=16,
        dyn_branch_ch=16,
        mask_head_ch=32,
        # c1_in_channels=64,
        # c1_channels=48,
        # in_channels=512,
        # channels=128,
        norm_cfg=norm_cfg,
        # pad_out_channel_factor=128/12,
        sampler=dict(type='OHEMPixelSampler', thresh=0.7, min_kept=100000)
    ),
    auxiliary_head=None,
)
