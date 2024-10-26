# Copyright (c) Wubiao Huang (https://github.com/HuangWBill).

# dataset settings
Potsdam_type = 'ISPRS_my_dataset'
Potsdam_root = 'data/Potsdam_IRRG_tif_512'
Potsdam_crop_size = (512, 512)

Potsdam_test_pipeline = [
    dict(type='LoadSingleRSImageFromFile'),
    dict(type='Resize', scale=(512, 512), keep_ratio=True),
    # add loading annotation after ``Resize`` because ground truth
    # does not need to do resize data transform
    dict(type='LoadAnnotations', reduce_zero_label=False),
    dict(type='PackSegInputs')
]

val_Potsdam = dict(
        type=Potsdam_type,
        data_root=Potsdam_root,
        data_prefix=dict(img_path='img_dir/val', seg_map_path='ann_dir/val'),
        pipeline=Potsdam_test_pipeline)

