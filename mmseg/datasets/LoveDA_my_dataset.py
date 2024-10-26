# Copyright (c) Wubiao Huang (https://github.com/HuangWBill).

from mmseg.registry import DATASETS
from mmseg.datasets.basesegdataset import BaseSegDataset

@DATASETS.register_module()
class LoveDA_my_dataset(BaseSegDataset):
    METAINFO = dict(
        classes=('background', 'building', 'road', 'water', 'barren', 'forest',
                 'agricultural'),
        palette=[[255, 255, 255], [255, 0, 0], [255, 255, 0], [0, 0, 255],
                 [159, 129, 183], [0, 255, 0], [255, 195, 128]])

    def __init__(self,
                 img_suffix='.png',
                 seg_map_suffix='.png',
                 reduce_zero_label=True,
                 **kwargs) -> None:
        super().__init__(
            img_suffix=img_suffix,
            seg_map_suffix=seg_map_suffix,
            reduce_zero_label=reduce_zero_label,
            **kwargs)
