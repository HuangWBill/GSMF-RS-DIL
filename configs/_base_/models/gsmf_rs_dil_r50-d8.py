# Copyright (c) Wubiao Huang (https://github.com/HuangWBill).

# model settings
norm_cfg_old = dict(type='BN', requires_grad=False)
norm_cfg = dict(type='BN', requires_grad=True)
data_preprocessor = dict(
    type='SegDataPreProcessor',
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    pad_val=0,
    seg_pad_val=255)
model = dict(
    type='DIL_EncoderDecoder',
    data_preprocessor=data_preprocessor,
    pretrained=None,
    DIL_type='LwF',
    backbone=dict(
        type='ResNetV1c',
        depth=50,
        in_channels=3,
        num_stages=4,
        base_channels=64,
        out_indices=(0, 1, 2, 3),
        dilations=(1, 1, 1, 1),
        strides=(1, 2, 2, 2),
        norm_cfg=norm_cfg_old,
        norm_eval=False,
        style='pytorch',
        contract_dilation=True),
    backbone_1=dict(
        type='ResNetV1c',
        depth=50,
        in_channels=3,
        num_stages=4,
        base_channels=64,
        out_indices=(0, 1, 2, 3),
        dilations=(1, 1, 1, 1),
        strides=(1, 2, 2, 2),
        norm_cfg=norm_cfg,
        norm_eval=False,
        style='pytorch',
        contract_dilation=True),
    decode_head=dict(
        type='GSMF_RS_DIL_Head',
        in_channels=[256, 512, 1024, 2048],
        in_index=[0, 1, 2, 3],
        channels=256,
        dropout_ratio=0.1,
        num_classes=6,
        mid_channels=256,
        N=64,
        reduction=8,
        norm_cfg=norm_cfg_old,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    decode_head_1=dict(
        type='GSMF_RS_DIL_Head',
        in_channels=[256, 512, 1024, 2048],
        in_index=[0, 1, 2, 3],
        channels=256,
        dropout_ratio=0.1,
        num_classes=6,
        mid_channels=256,
        N=64,
        reduction=8,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))
