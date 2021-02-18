_base_ = [
    '../_base_/models/padnet_r50.py', '../_base_/datasets/cityscapes.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_40k.py'
]

model = dict(
    use_aligned_bilinear=True,

    decode_head=dict(
        upsample_factor=16,
        dyn_branch_ch=8,
        mask_head_ch=16,
        sem_loss_on=True,
        use_aligned_bilinear=True,
    ),
    auxiliary_head=None
)
