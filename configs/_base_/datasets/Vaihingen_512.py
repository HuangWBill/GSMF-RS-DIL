# Copyright (c) Wubiao Huang (https://github.com/HuangWBill).

# dataset settings
Vaihingen_type = 'ISPRS_my_dataset'
Vaihingen_root = 'data/Vaihingen_IRRG_tif_512'
Vaihingen_crop_size = (512, 512)

Vaihingen_train_pipeline = [
    dict(type='LoadSingleRSImageFromFile'),
    dict(type='LoadAnnotations', reduce_zero_label=False),
    dict(
        type='RandomResize',
        scale=(512, 512),
        ratio_range=(0.5, 2.0),
        keep_ratio=True),
    dict(type='RandomCrop', crop_size=Vaihingen_crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackSegInputs')
]

train_Vaihingen = dict(
        type=Vaihingen_type,
        data_root=Vaihingen_root,
        data_prefix=dict(
            img_path='img_dir/train', seg_map_path='ann_dir/train'),
        pipeline=Vaihingen_train_pipeline)

Vaihingen_test_pipeline = [
    dict(type='LoadSingleRSImageFromFile'),
    dict(type='Resize', scale=(512, 512), keep_ratio=True),
    # add loading annotation after ``Resize`` because ground truth
    # does not need to do resize data transform
    dict(type='LoadAnnotations', reduce_zero_label=False),
    dict(type='PackSegInputs')
]

val_Vaihingen = dict(
        type=Vaihingen_type,
        data_root=Vaihingen_root,
        data_prefix=dict(img_path='img_dir/val', seg_map_path='ann_dir/val'),
        pipeline=Vaihingen_test_pipeline)

