_base_ = [
    '../_base_/models/padnet_r50.py', '../_base_/datasets/ade20k.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_80k.py'
]

model = dict(
    use_aligned_bilinear=True,

    decode_head=dict(
        upsample_factor=16,
        dyn_branch_ch=8,
        mask_head_ch=16,
        sem_loss_on=True,
        use_aligned_bilinear=True,
        tower_ch=128,
        tower_num_convs=4,
        num_classes=150,
    ),
    auxiliary_head=None
)
