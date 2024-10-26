# Copyright (c) Wubiao Huang (https://github.com/HuangWBill).
# GSMF-RS-DIL
_base_ = [
    '../_base_/models/gsmf_rs_dil_r50-d8.py', '../_base_/datasets/ISPRS_DIL_my_dataset_IRRG_512.py', '../_base_/default_runtime_iter.py'
]

custom_imports = dict(imports=['projects.DIL.mmseg.datasets.ISPRS_my_dataset',
                               'projects.DIL.mmseg.decode_head.gsmf_rs_dil_head',
                               'projects.DIL.mmseg.evaluation.dg_metrics',
                               'projects.DIL.mmseg.segmentors.DIL_encoder_decoder',
                               'projects.DIL.mmseg.losses.dil_hin_loss',
                               'projects.DIL.mmseg.losses.dil_kldivergence_loss'
                               ])

crop_size = (512,512)
data_preprocessor = dict(size=crop_size)
model = dict(type='DIL_EncoderDecoder',
             data_preprocessor=data_preprocessor,
             DIL_type='GSMF-RS',
             frozen_backbone_1=True,
             frozen_backbone_layer=0,
             frozen_decode_1=False,
             frozen_decode_layer=-1,
             backbone=dict(depth=101),
             backbone_1=dict(depth=101),
             decode_head=dict(type='GSMF_RS_DIL_Head',num_classes=6,mid_channels=128, N=64, reduction=16,kernel_size=3),
             decode_head_1=dict(type='GSMF_RS_DIL_Head',num_classes=6,mid_channels=128, N=64, reduction=16,kernel_size=3,
                                loss_decode=[dict(type='CrossEntropyLoss', use_sigmoid=False, loss_name='loss_ce', loss_weight=1.0),
                                             dict(type='DIL_HinLoss', temperature=2.0, loss_name='loss_dilhin', loss_weight=0.5),
                                             dict(type='DIL_KLDivLoss', temperature=1.0, idx=[1,2,3], loss_name='loss_kld', loss_weight=1.0)]
             ))

load_from='result/GS-AUFPN-Potsdam/iter_80000_DIL.pth'

randomness =dict(seed=0)

# optimizer
optimizer=dict(type='AdamW', lr=0.00001, betas=(0.9, 0.999), weight_decay=0.001)
optim_wrapper = dict(type='OptimWrapper', optimizer=optimizer, clip_grad=None)

# learning policy
param_scheduler = [
    dict(
        type='PolyLR',
        eta_min=0.0,
        power=0.9,
        begin=0,
        end=10000,
        by_epoch=False)
]

# training schedule for 10k
train_cfg = dict(type='IterBasedTrainLoop', max_iters=10000, val_interval=200)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=1, log_metric_by_epoch=False),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', by_epoch=False, interval=200),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='SegVisualizationHook'))



