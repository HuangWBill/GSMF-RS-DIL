# Copyright (c) Wubiao Huang (https://github.com/HuangWBill).

# dataset settings
LoveDA_Urban_type = 'LoveDA_my_dataset'
LoveDA_Urban_root = 'data/LoveDA_Urban_512'
LoveDA_Urban_crop_size = (512, 512)

LoveDA_Urban_test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(512, 512), keep_ratio=True),
    # add loading annotation after ``Resize`` because ground truth
    # does not need to do resize data transform
    dict(type='LoadAnnotations', reduce_zero_label=True),
    dict(type='PackSegInputs')
]

val_LoveDA_Urban = dict(
        type=LoveDA_Urban_type,
        data_root=LoveDA_Urban_root,
        data_prefix=dict(img_path='img_dir/val', seg_map_path='ann_dir/val'),
        pipeline=LoveDA_Urban_test_pipeline)

