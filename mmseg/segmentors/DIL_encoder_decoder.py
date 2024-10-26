# Copyright (c) Wubiao Huang (https://github.com/HuangWBill).

from typing import List, Optional
import torch
import torch.nn.functional as F
from torch import Tensor
from mmseg.registry import MODELS
from mmseg.utils import (ConfigType, OptConfigType, OptMultiConfig,
                         OptSampleList, SampleList, add_prefix)
from mmseg.models.segmentors.base import BaseSegmentor

@MODELS.register_module()
class DIL_EncoderDecoder(BaseSegmentor):
    def __init__(self,
                 backbone: ConfigType,
                 backbone_1: ConfigType,
                 decode_head: ConfigType,
                 decode_head_1: ConfigType,
                 DIL_type,
                 frozen_backbone_1=False,
                 frozen_backbone_layer=-1,
                 frozen_decode_1=False,
                 frozen_decode_layer=-1,
                 neck: OptConfigType = None,
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 data_preprocessor: OptConfigType = None,
                 pretrained: Optional[str] = None,
                 init_cfg: OptMultiConfig = None):
        super().__init__(
            data_preprocessor=data_preprocessor, init_cfg=init_cfg)
        if pretrained is not None:
            backbone.init_cfg = dict(type='Pretrained_Part', checkpoint=pretrained)
            backbone_1.init_cfg = dict(type='Pretrained_Part', checkpoint=pretrained)

        self.backbone = MODELS.build(backbone)
        self.backbone_1 = MODELS.build(backbone_1)
        if neck is not None:
            self.neck = MODELS.build(neck)
        self._init_decode_head(decode_head)
        self._init_decode_head_1(decode_head_1)
        self.DIL_type=DIL_type
        self.frozen_backbone_1=frozen_backbone_1
        self.frozen_backbone_layer=frozen_backbone_layer
        self.frozen_decode_1=frozen_decode_1
        self.frozen_decode_layer = frozen_decode_layer
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        if self.frozen_backbone_layer == 0:
            self.frozen_backbone_name=['stem','layer1']
        elif self.frozen_backbone_layer == 1:
            self.frozen_backbone_name=['stem','layer1','layer2']
        elif self.frozen_backbone_layer == 2:
            self.frozen_backbone_name=['stem','layer1','layer2','layer3']
        else:
            self.frozen_backbone_name = None

        if self.frozen_decode_layer == 0:
            self.frozen_decode_name=['conv_seg.']
        elif self.frozen_decode_layer == 1:
            self.frozen_decode_name=['conv_seg.','cam.','sam.']
        else:
            self.frozen_decode_name = None

        assert self.with_decode_head

    def _init_decode_head(self, decode_head: ConfigType) -> None:
        """Initialize ``decode_head``"""
        self.decode_head = MODELS.build(decode_head)
        self.align_corners = self.decode_head.align_corners
        self.num_classes = self.decode_head.num_classes
        self.out_channels = self.decode_head.out_channels

    def _init_decode_head_1(self, decode_head: ConfigType) -> None:
        """Initialize ``decode_head``"""
        self.decode_head_1 = MODELS.build(decode_head)
        self.align_corners = self.decode_head_1.align_corners
        self.num_classes = self.decode_head_1.num_classes
        self.out_channels = self.decode_head_1.out_channels

    def extract_feat(self, inputs: Tensor):
        """Extract visual features from images."""
        for param in self.backbone.parameters():
            param.requires_grad = False
        x = self.backbone(inputs)
        if self.frozen_backbone_1==True:
            for name, param in self.backbone_1.named_parameters():
                if self.frozen_backbone_name==None:
                    param.requires_grad = False
                else:
                    for layers in self.frozen_backbone_name:
                        if layers in name:
                            param.requires_grad = False

        x_1 = self.backbone_1(inputs)
        if self.with_neck:
            x_n1 = self.neck(x_1)
        else:
            x_n1=x_1
        return x, x_1, x_n1

    def encode_decode(self, inputs: Tensor,
                      batch_img_metas: List[dict]) -> Tensor:
        """Encode the name of classes with text_encoder and encode images with
        image_encoder.

        Then decode the class embedding and visual feature into a semantic
        segmentation map of the same size as input.
        """
        x = self.backbone_1(inputs)
        if self.with_neck:
            x = self.neck(x)
        seg_logits = self.decode_head_1.predict(x, batch_img_metas,self.test_cfg)

        return seg_logits

    def _decode_head_forward_train(self, new_x: List[Tensor], old_y: Tensor,seg_labels: Tensor,old_x=None,new_da_x=None) -> dict:
        """Run forward function and calculate loss for decode head in
        training."""
        losses = dict()

        if self.frozen_decode_1==True:
            for name, param in self.decode_head_1.named_parameters():
                if self.frozen_decode_name==None:
                    param.requires_grad = False
                else:
                    for layers in self.frozen_decode_name:
                        if layers not in name:
                            param.requires_grad = False

        loss_decode = self.decode_head_1.loss(new_x, old_y, seg_labels, old_x=old_x,new_da_x=new_da_x)
        losses.update(add_prefix(loss_decode, 'decode'))
        return losses

    def _stack_batch_gt(self, batch_data_samples: SampleList) -> Tensor:
        gt_semantic_segs = [data_sample.gt_sem_seg.data for data_sample in batch_data_samples]
        return torch.stack(gt_semantic_segs, dim=0)

    def loss(self, inputs: Tensor, data_samples: SampleList) -> dict:
        """Calculate losses from a batch of inputs and data samples.

        Args:
            inputs (Tensor): Input images.
            data_samples (list[:obj:`SegDataSample`]): The seg data samples.
                It usually includes information such as `metainfo` and
                `gt_sem_seg`.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        old_x, new_x, new_da_x = self.extract_feat(inputs)
        losses = dict()
        for param in self.decode_head.parameters():
            param.requires_grad = False
        old_y = self.decode_head.predict_foward(old_x)
        seg_label = self._stack_batch_gt(data_samples)

        if self.DIL_type == 'LwF':
            loss_decode = self._decode_head_forward_train(new_x, old_y, seg_label)
        elif self.DIL_type == 'ours':
            loss_decode = self._decode_head_forward_train(new_x, old_y, seg_label, old_x=old_x, new_da_x=new_da_x)
        else:
            loss_decode = self._decode_head_forward_train(new_x, old_y, seg_label, old_x=old_x)

        losses.update(loss_decode)
        return losses

    def predict(self,
                inputs: Tensor,
                data_samples: OptSampleList = None) -> SampleList:
        """Predict results from a batch of inputs and data samples with post-
        processing.

        Args:
            inputs (Tensor): Inputs with shape (N, C, H, W).
            data_samples (List[:obj:`SegDataSample`], optional): The seg data
                samples. It usually includes information such as `metainfo`
                and `gt_sem_seg`.

        Returns:
            list[:obj:`SegDataSample`]: Segmentation results of the
            input images. Each SegDataSample usually contain:

            - ``pred_sem_seg``(PixelData): Prediction of semantic segmentation.
            - ``seg_logits``(PixelData): Predicted logits of semantic
                segmentation before normalization.
        """
        if data_samples is not None:
            batch_img_metas = [
                data_sample.metainfo for data_sample in data_samples
            ]
        else:
            batch_img_metas = [
                dict(
                    ori_shape=inputs.shape[2:],
                    img_shape=inputs.shape[2:],
                    pad_shape=inputs.shape[2:],
                    padding_size=[0, 0, 0, 0])
            ] * inputs.shape[0]

        seg_logits = self.inference(inputs, batch_img_metas)

        return self.postprocess_result(seg_logits, data_samples)

    def _forward(self,
                 inputs: Tensor,
                 data_samples: OptSampleList = None) -> Tensor:
        """Network forward process.

        Args:
            inputs (Tensor): Inputs with shape (N, C, H, W).
            data_samples (List[:obj:`SegDataSample`]): The seg
                data samples. It usually includes information such
                as `metainfo` and `gt_sem_seg`.

        Returns:
            Tensor: Forward output of model without any post-processes.
        """
        x,x_1,x_n1 = self.extract_feat(inputs)
        return self.decode_head_1.forward(x_1)

    def slide_inference(self, inputs: Tensor,
                        batch_img_metas: List[dict]) -> Tensor:
        """Inference by sliding-window with overlap.

        If h_crop > h_img or w_crop > w_img, the small patch will be used to
        decode without padding.

        Args:
            inputs (tensor): the tensor should have a shape NxCxHxW,
                which contains all images in the batch.
            batch_img_metas (List[dict]): List of image metainfo where each may
                also contain: 'img_shape', 'scale_factor', 'flip', 'img_path',
                'ori_shape', and 'pad_shape'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:PackSegInputs`.

        Returns:
            Tensor: The segmentation results, seg_logits from model of each
                input image.
        """

        h_stride, w_stride = self.test_cfg.stride
        h_crop, w_crop = self.test_cfg.crop_size
        batch_size, _, h_img, w_img = inputs.size()
        out_channels = self.out_channels
        h_grids = max(h_img - h_crop + h_stride - 1, 0) // h_stride + 1
        w_grids = max(w_img - w_crop + w_stride - 1, 0) // w_stride + 1
        preds = inputs.new_zeros((batch_size, out_channels, h_img, w_img))
        count_mat = inputs.new_zeros((batch_size, 1, h_img, w_img))
        for h_idx in range(h_grids):
            for w_idx in range(w_grids):
                y1 = h_idx * h_stride
                x1 = w_idx * w_stride
                y2 = min(y1 + h_crop, h_img)
                x2 = min(x1 + w_crop, w_img)
                y1 = max(y2 - h_crop, 0)
                x1 = max(x2 - w_crop, 0)
                crop_img = inputs[:, :, y1:y2, x1:x2]
                # change the image shape to patch shape
                batch_img_metas[0]['img_shape'] = crop_img.shape[2:]
                # the output of encode_decode is seg logits tensor map
                # with shape [N, C, H, W]
                crop_seg_logit = self.encode_decode(crop_img, batch_img_metas)
                preds += F.pad(crop_seg_logit,
                               (int(x1), int(preds.shape[3] - x2), int(y1),
                                int(preds.shape[2] - y2)))

                count_mat[:, :, y1:y2, x1:x2] += 1
        assert (count_mat == 0).sum() == 0
        seg_logits = preds / count_mat

        return seg_logits

    def whole_inference(self, inputs: Tensor, batch_img_metas: List[dict]) -> Tensor:
        """Inference with full image.

        Args:
            inputs (Tensor): The tensor should have a shape NxCxHxW, which
                contains all images in the batch.
            batch_img_metas (List[dict]): List of image metainfo where each may
                also contain: 'img_shape', 'scale_factor', 'flip', 'img_path',
                'ori_shape', and 'pad_shape'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:PackSegInputs`.

        Returns:
            Tensor: The segmentation results, seg_logits from model of each
                input image.
        """

        seg_logits = self.encode_decode(inputs, batch_img_metas)

        return seg_logits

    def inference(self, inputs: Tensor, batch_img_metas: List[dict]) -> Tensor:
        """Inference with slide/whole style.

        Args:
            inputs (Tensor): The input image of shape (N, 3, H, W).
            batch_img_metas (List[dict]): List of image metainfo where each may
                also contain: 'img_shape', 'scale_factor', 'flip', 'img_path',
                'ori_shape', 'pad_shape', and 'padding_size'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:PackSegInputs`.

        Returns:
            Tensor: The segmentation results, seg_logits from model of each
                input image.
        """

        assert self.test_cfg.mode in ['slide', 'whole']
        ori_shape = batch_img_metas[0]['ori_shape']
        assert all(_['ori_shape'] == ori_shape for _ in batch_img_metas)
        if self.test_cfg.mode == 'slide':
            seg_logit = self.slide_inference(inputs, batch_img_metas)
        else:
            seg_logit = self.whole_inference(inputs, batch_img_metas)

        return seg_logit

    def aug_test(self, inputs, batch_img_metas, rescale=True):
        """Test with augmentations.

        Only rescale=True is supported.
        """
        # aug_test rescale all imgs back to ori_shape for now
        assert rescale
        # to save memory, we get augmented seg logit inplace
        seg_logit = self.inference(inputs[0], batch_img_metas[0], rescale)
        for i in range(1, len(inputs)):
            cur_seg_logit = self.inference(inputs[i], batch_img_metas[i], rescale)
            seg_logit += cur_seg_logit
        seg_logit /= len(inputs)
        seg_pred = seg_logit.argmax(dim=1)
        # unravel batch dim
        seg_pred = list(seg_pred)
        return seg_pred
