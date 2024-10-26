# Copyright (c) Wubiao Huang (https://github.com/HuangWBill).

# dataset settings
LoveDA_Rural_type = 'LoveDA_my_dataset'
LoveDA_Rural_root = 'data/LoveDA_Rural_512'
LoveDA_Rural_crop_size = (512, 512)

LoveDA_Rural_train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', reduce_zero_label=True),
    dict(
        type='RandomResize',
        scale=(512, 512),
        ratio_range=(0.5, 2.0),
        keep_ratio=True),
    dict(type='RandomCrop', crop_size=LoveDA_Rural_crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackSegInputs')
]

train_LoveDA_Rural = dict(
        type=LoveDA_Rural_type,
        data_root=LoveDA_Rural_root,
        data_prefix=dict(
            img_path='img_dir/train', seg_map_path='ann_dir/train'),
        pipeline=LoveDA_Rural_train_pipeline)

LoveDA_Rural_test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(512, 512), keep_ratio=True),
    # add loading annotation after ``Resize`` because ground truth
    # does not need to do resize data transform
    dict(type='LoadAnnotations', reduce_zero_label=True),
    dict(type='PackSegInputs')
]

val_LoveDA_Rural = dict(
        type=LoveDA_Rural_type,
        data_root=LoveDA_Rural_root,
        data_prefix=dict(img_path='img_dir/val', seg_map_path='ann_dir/val'),
        pipeline=LoveDA_Rural_test_pipeline)
