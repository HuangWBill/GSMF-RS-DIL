# Copyright (c) Wubiao Huang (https://github.com/HuangWBill).

# dataset settings
multi_type = 'LoveDA_my_dataset'
multi_root = 'data/LoveDA_all_512'
multi_crop_size = (512, 512)
multi_train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', reduce_zero_label=True),
    dict(
        type='RandomResize',
        scale=(512, 512),
        ratio_range=(0.5, 2.0),
        keep_ratio=True),
    dict(type='RandomCrop', crop_size=multi_crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackSegInputs')
]

train_multi = dict(
        type=multi_type,
        data_root=multi_root,
        data_prefix=dict(
            img_path='img_dir/train', seg_map_path='ann_dir/train'),
        pipeline=multi_train_pipeline)
