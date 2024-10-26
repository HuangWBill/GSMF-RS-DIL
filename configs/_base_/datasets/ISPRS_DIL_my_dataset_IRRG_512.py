# Copyright (c) Wubiao Huang (https://github.com/HuangWBill).
_base_ = [
    "./Potsdam_IRRG_512.py",
    "./Vaihingen_512.py",
]

train_dataloader = dict(
    batch_size=4,
    num_workers=1,
    persistent_workers=True,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    dataset={{_base_.train_Vaihingen}})

val_dataloader = dict(
    batch_size=1,
    num_workers=1,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type="ConcatDataset",
        datasets=[{{_base_.val_Potsdam}}, {{_base_.val_Vaihingen}},]
        ))
test_dataloader = val_dataloader

val_evaluator = dict(
    type="DGIoUMetric",
    iou_metrics=["mIoU"],
    dataset_keys=["Potsdam", "Vaihingen"],
)

test_evaluator = val_evaluator