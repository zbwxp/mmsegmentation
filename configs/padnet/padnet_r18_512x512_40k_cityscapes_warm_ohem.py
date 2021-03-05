_base_ = [
    '../_base_/models/padnet_r50.py', '../_base_/datasets/cityscapes_512x512.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_40k_warmup.py'
]
norm_cfg = dict(type='BN', requires_grad=True)
model = dict(
    pretrained='open-mmlab://resnet18_v1c',
    use_aligned_bilinear=True,
    backbone=dict(
        depth=18,
        norm_cfg=norm_cfg,
    ),
    decode_head=dict(
        upsample_factor=16,
        dyn_branch_ch=8,
        mask_head_ch=8,
        sem_loss_on=True,
        use_aligned_bilinear=True,
        in_channels=(64, 128, 256, 512),
        channels=1, # this channel is not used
        norm_cfg=norm_cfg,
        sampler=dict(type='OHEMPixelSampler', thresh=0.7, min_kept=100000)
    ),
    auxiliary_head=None
)
