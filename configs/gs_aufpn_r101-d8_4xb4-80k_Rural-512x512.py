# Copyright (c) Wubiao Huang (https://github.com/HuangWBill).
# gs_aufpn-ResNet101
_base_ = [
    '../_base_/models/gs_aufpn_r50-d8.py', '../_base_/datasets/LoveDA_Rural_my_dataset_512.py', '../_base_/default_runtime_iter.py',
    '../_base_/schedules/schedule_30k_iter.py'
]

custom_imports = dict(imports=['GSMF-RS-DIL.mmseg.datasets.LoveDA_my_dataset','GSMF-RS-DIL.mmseg.decode_head.gs_aufpn_head'])

crop_size = (512,512)
data_preprocessor = dict(size=crop_size)
model = dict(data_preprocessor=data_preprocessor,
             pretrained='open-mmlab://resnet101_v1c',
             backbone=dict(depth=101),
             decode_head=dict(num_classes=7,mid_channels=128, N=64, reduction=16,kernel_size=3),
             auxiliary_head=dict(num_classes=7))
