# Copyright (c) Wubiao Huang (https://github.com/HuangWBill).

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmseg.registry import MODELS

@MODELS.register_module()
class DIL_HinLoss(nn.Module):
    def __init__(
        self,
        temperature: float = 1.0,
        reduction: str = 'mean',
        idx=[3],
        idx_type='mean',
        loss_weight: float = 1.0,
        loss_name: str = 'loss_dilhin') -> None:
        assert isinstance(temperature, (float, int)), \
            'Expected temperature to be' \
            f'float or int, but got {temperature.__class__.__name__} instead'
        assert temperature != 0., 'Temperature must not be zero'

        assert reduction in ['mean', 'none', 'sum'], \
            'Reduction must be one of the options ("mean", ' \
            f'"sum", "none"), but got {reduction}'
        super().__init__()
        self.temperature = temperature
        self.loss_weight = loss_weight
        self.reduction = reduction
        self.idx=idx
        self.idx_type = idx_type
        self._loss_name = loss_name

    def forward(self, preds_new, preds_old,weight=None,ignore_index=255):
        """Forward function.
        Args:
            preds_new (torch.Tensor): Logit tensor,
                the data type is float32 or float64.
                The shape is (N, C, H, W) or shape (N, C).
           preds_old (torch.Tensor): Logit tensor,
                the data type is float32 or float64.
                The shape is (N, C, H, W) or shape (N, C).
                preds_new and preds_old must be with the same shape.

        Returns:
            (Tensor): Reduced loss.
        """
        if isinstance(preds_new, torch.Tensor):
            assert isinstance(preds_old, torch.Tensor), 'Expected target to' \
                                                        f'be Tensor, but got {preds_old.__class__.__name__} instead'

            assert preds_new.shape == preds_old.shape, 'Input and target ' \
                                                    'must have same shape,' \
                                                    f'but got shapes {preds_new.shape} and {preds_old.shape}'

            preds_new = F.log_softmax(preds_new / self.temperature, dim=1)
            preds_old = F.softmax(preds_old / self.temperature, dim=1)

            outputs = torch.sum(preds_new * preds_old, dim=1, keepdim=False)
            loss = -torch.mean(outputs, dim=0, keepdim=False)
            loss = self.loss_weight * torch.mean(loss)
        else:
            assert len(preds_new) == len(preds_old), 'Input and target ' \
                                                       'must have same shape,' \
                                                       f'but got shapes {len(preds_new)} and {len(preds_old)}'
            assert preds_new[-1].shape == preds_old[-1].shape, 'Input and target ' \
                                                       'must have same shape,' \
                                                       f'but got shapes {preds_new[-1].shape} and {preds_old[-1].shape}'
            loss_=0
            for idx in self.idx:
                preds_new_ = F.log_softmax(preds_new[idx] / self.temperature, dim=1)
                preds_old_ = F.softmax(preds_old[idx] / self.temperature, dim=1)

                outputs = torch.sum(preds_new_ * preds_old_, dim=1, keepdim=False)
                loss = torch.mean(-torch.mean(outputs, dim=0, keepdim=False))
                loss_=loss_+loss
            if self.idx_type == 'sum':
                loss = self.loss_weight * loss_
            else:
                loss = self.loss_weight * (loss_ / len(self.idx))
        return loss

    @property
    def loss_name(self):
        """Loss Name.

        This function must be implemented and will return the name of this
        loss function. This name will be used to combine different loss items
        by simple sum operation. In addition, if you want this loss item to be
        included into the backward graph, `loss_` must be the prefix of the
        name.

        Returns:
            str: The name of this loss item.
        """
        return self._loss_name