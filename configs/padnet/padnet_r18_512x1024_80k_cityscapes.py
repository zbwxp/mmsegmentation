_base_ = './padnet_r50_512x1024_80k_cityscapes.py'
model = dict(
    pretrained='open-mmlab://resnet18_v1c',
    use_aligned_bilinear=True,
    backbone=dict(depth=18),
    decode_head=dict(
        upsample_factor=16,
        dyn_branch_ch=8,
        mask_head_ch=8,
        sem_loss_on=True,
        use_aligned_bilinear=True,
        in_channels=(64, 128, 256, 512),
        channels=1, # this channel is not used
    ),
    auxiliary_head=None
)
