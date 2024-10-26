# Copyright (c) Wubiao Huang (https://github.com/HuangWBill).

_base_ = [
    "./LoveDA_Urban_512.py",
    "./LoveDA_Rural_512.py",
]

train_dataloader = dict(
    batch_size=4,
    num_workers=1,
    persistent_workers=True,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    dataset={{_base_.train_LoveDA_Rural}})
val_dataloader = dict(
    batch_size=1,
    num_workers=1,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type="ConcatDataset",
        datasets=[{{_base_.val_LoveDA_Urban}}, {{_base_.val_LoveDA_Rural}},]
        ))

test_dataloader = val_dataloader

val_evaluator = dict(
    type="DGIoUMetric",
    iou_metrics=["mIoU"],
    dataset_keys=["LoveDA_Urban", "LoveDA_Rural"],
)

test_evaluator = val_evaluator
