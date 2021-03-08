_base_ = [
    '../_base_/models/padnet_r50.py', '../_base_/datasets/cityscapes_512x512.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_40k_warmup.py'
]
norm_cfg = dict(type='BN', requires_grad=True)
model = dict(
    use_aligned_bilinear=True,
    backbone=dict(
        norm_cfg=norm_cfg,
    ),
    decode_head=dict(
        upsample_factor=16,
        dyn_branch_ch=16,
        mask_head_ch=32,
        sem_loss_on=False,
        use_aligned_bilinear=True,
        tower_ch=128,
        # in_channels=(64, 128, 256, 512),
        norm_cfg=norm_cfg,
        sampler=dict(type='OHEMPixelSampler', thresh=0.7, min_kept=100000)
    ),
    auxiliary_head=dict(norm_cfg=norm_cfg)
)
