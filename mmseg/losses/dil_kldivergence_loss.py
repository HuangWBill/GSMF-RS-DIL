# Copyright (c) Wubiao Huang (https://github.com/HuangWBill).

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmseg.registry import MODELS

@MODELS.register_module()
class DIL_KLDivLoss(nn.Module):
    def __init__(self,
                 temperature: float = 1.0,
                 reduction: str = 'mean',
                 idx=[3],
                 idx_type='mean',
                 loss_weight: float = 1.0,
                 loss_name: str = 'loss_kld'):

        assert isinstance(temperature, (float, int)), \
            'Expected temperature to be' \
            f'float or int, but got {temperature.__class__.__name__} instead'
        assert temperature != 0., 'Temperature must not be zero'

        assert reduction in ['mean', 'batchmean','none', 'sum'], \
            'Reduction must be one of the options ("mean", ' \
            f'"sum", "batchmean","none"), but got {reduction}'

        super().__init__()
        self.temperature = temperature
        self.reduction = reduction
        self.idx = idx
        self.idx_type = idx_type
        self.loss_weight = loss_weight
        self._loss_name = loss_name

    def forward(self, preds_new, preds_old,weight=None,ignore_index=255):
        """Forward function. Calculate KL divergence Loss.
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
            softmax_preds_old = F.softmax(preds_old / self.temperature, dim=1)
            logsoftmax_preds_new = F.log_softmax(preds_new / self.temperature, dim=1)
            loss = (self.temperature ** 2) * F.kl_div(logsoftmax_preds_new, softmax_preds_old,reduction='none')
            if self.reduction == 'sum':
                # Change view to calculate instance-wise sum
                loss = loss.view(preds_old.shape[0], -1)
                loss = torch.sum(loss, dim=1)
            elif self.reduction == 'mean':
                # Change view to calculate instance-wise mean
                loss = loss.view(preds_old.shape[0], -1)
                loss = torch.mean(loss, dim=1)
            loss = self.loss_weight * loss
        else:
            assert len(preds_new) == len(preds_old), 'Input and target ' \
                                                     'must have same shape,' \
                                                     f'but got shapes {len(preds_new)} and {len(preds_old)}'
            assert preds_new[-1].shape == preds_old[-1].shape, 'Input and target ' \
                                                               'must have same shape,' \
                                                               f'but got shapes {preds_new[-1].shape} and {preds_old[-1].shape}'
            loss_ = 0
            for idx in self.idx:
                softmax_preds_old = F.softmax(preds_old[idx] / self.temperature, dim=1)
                logsoftmax_preds_new = F.log_softmax(preds_new[idx] / self.temperature, dim=1)
                loss = (self.temperature ** 2) * F.kl_div(logsoftmax_preds_new, softmax_preds_old,reduction='none')
                if self.reduction == 'sum':
                    # Change view to calculate instance-wise sum
                    loss = loss.view(preds_old[idx].shape[0], -1)
                    loss = torch.sum(loss, dim=1)
                elif self.reduction == 'mean':
                    # Change view to calculate instance-wise mean
                    loss = loss.view(preds_old[idx].shape[0], -1)
                    loss = torch.mean(loss, dim=1)
                loss_ = loss_ + loss
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
