# Copyright (c) Wubiao Huang (https://github.com/HuangWBill).

# dataset settings
multi_type = 'ISPRS_my_dataset'
multi_root = 'data/ISPRS_IRRG_512'
multi_crop_size = (512, 512)

multi_train_pipeline = [
    dict(type='LoadSingleRSImageFromFile'),
    dict(type='LoadAnnotations', reduce_zero_label=False),
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



