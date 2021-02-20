_base_ = './pad_ppmnet_r50_1024x1024_80k_cityscapes.py'
model = dict(
    pretrained='open-mmlab://resnet18_v1c',
    use_aligned_bilinear=True,
    backbone=dict(depth=18),
    decode_head=dict(
        upsample_factor=8,
        dyn_branch_ch=8,
        mask_head_ch=8,
        sem_loss_on=False,  # Do Not use sem_loss for now
        use_aligned_bilinear=True,
        in_channels=(64, 128, 256, 512),
        channels=256,  # 256 for R18
    ),
    auxiliary_head=dict(in_channels=256, channels=64)
)
