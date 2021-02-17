# model settings
norm_cfg = dict(type='BN', requires_grad=True)  # default 'SyncBN'
model = dict(
    type='EncoderDecoder',
    pretrained='open-mmlab://resnet50_v1c',
    backbone=dict(
        type='ResNetV1c',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        dilations=(1, 1, 1, 1),
        strides=(1, 2, 2, 2),
        norm_cfg=norm_cfg,
        norm_eval=False,
        style='pytorch',
        contract_dilation=True),
    decode_head=dict(
        type='DynConvHead',
        upsample_factor=16,
        dyn_branch_ch=8,
        mask_head_ch=16,
        use_low_level_info=True,
        low_level_stages=(0, 1, 2),  # starts from stage_2
        tower_ch=48,
        in_channels=(256, 512, 1024, 2048),
        input_transform='multiple_select',
        in_index=(0, 1, 2, 3),
        channels=512,  # channels to cls 1x1 conv "self.conv_seg"
        dropout_ratio=0.1,
        num_classes=19,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    auxiliary_head=dict(
        type='FCNHead',
        in_channels=1024,
        in_index=2,
        channels=256,
        num_convs=1,
        concat_input=False,
        dropout_ratio=0.1,
        num_classes=19,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4)),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))
