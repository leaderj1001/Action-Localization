import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from torchvision.ops import RoIAlign, RoIPool, nms

import numpy as np

from utils.config import cfg
from nets.generate_anchors import generate_anchors_pre
from nets.proposal_layer import proposal_layer
from nets.anchor_target_layer import anchor_target_layer
from nets.proposal_target_layer import proposal_target_layer
from nets.bbox_transform import bbox_transform_inv, clip_boxes, bbox_overlaps
from models.pytorch_i3d import InceptionI3d
from models.i3dpt import I3D


class Transformer(nn.Module):
    def __init__(self, in_channels):
        super(Transformer, self).__init__()
        self.dropout_rate = 0.3
        self.hidden_size = 2048
        self.in_channels = in_channels

        self.dropout1 = nn.Dropout(self.dropout_rate)
        self.layer_norm1 = nn.LayerNorm(in_channels)

        self.dropout2 = nn.Dropout(self.dropout_rate)
        self.layer_norm2 = nn.LayerNorm(in_channels)

        self.fc1 = nn.Linear(in_channels, self.hidden_size)
        self.dropout3 = nn.Dropout(self.dropout_rate)
        self.fc2 = nn.Linear(self.hidden_size, in_channels)

        self.init_weights()

    def forward(self, roi_pool, video_feature):
        query = roi_pool.contiguous().view(-1, self.in_channels)

        key = video_feature.view(self.in_channels, -1)
        value = video_feature.view(-1, self.in_channels)

        out = torch.matmul(query, key) / float(pow(self.in_channels, 0.5))
        attn = F.softmax(out, dim=-1)

        out = torch.matmul(attn, value)

        out = self.layer_norm1(query + self.dropout1(out))
        out = self.fc2(F.relu(self.fc1(out)))
        out = self.layer_norm2(out + self.dropout2(out))

        return out.unsqueeze(dim=2).unsqueeze(dim=3)

    def init_weights(self):
        def normal_init(m, mean, stddev, truncated=False):
            """
            weight initalizer: truncated normal and random normal.
            """
            # x is a parameter
            if truncated:
                m.weight.data.normal_().fmod_(2).mul_(stddev).add_(mean)  # not a perfect approximation
            else:
                m.weight.data.normal_(mean, stddev)
            m.bias.data.zero_()

        normal_init(self.fc1, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.fc2, 0, 0.01, cfg.TRAIN.TRUNCATED)


class TransformerBlock(nn.Module):
    def __init__(self, in_channels):
        super(TransformerBlock, self).__init__()
        self.hidden_size = 128

        self.transformer1 = Transformer(self.hidden_size // 2)
        self.transformer2 = Transformer(self.hidden_size // 2)

        self.linear_projection_3d = nn.Conv3d(in_channels, self.hidden_size, kernel_size=(1, 1, 1))

    def forward(self, roi_pool, video_feature):
        projection_video_feature = self.linear_projection_3d(F.relu(video_feature))

        roi_pool1, roi_pool2 = torch.chunk(roi_pool, chunks=2, dim=1)
        video_feature1, video_feature2 = torch.chunk(projection_video_feature, chunks=2, dim=1)

        out1 = self.transformer1(roi_pool1, video_feature1)
        out2 = self.transformer2(roi_pool2, video_feature2)

        out = torch.cat((out1, out2), dim=1)

        return out


class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self._num_classes = 81
        self._device = 'cuda'
        self._layers = {}
        self._predictions = {}
        self._proposal_targets = {}
        self._anchor_targets = {}
        self._losses = {}

        self._check = False

        self._feat_stride = [
            16,
        ]
        self._feat_compress = [
            1. / float(self._feat_stride[0]),
        ]
        # self._net_conv_channels = 1024
        self._net_conv_channels = 832
        self._linear_projection_channels = 128

    def _init_head_tail(self, pretrained='Kinetics'):
        if pretrained == 'ImageNet':
            self.i3d = InceptionI3d()
        elif pretrained == 'Kinetics':
            self.i3d = I3D(num_classes=400, modality='rgb')

        for param in self.i3d.parameters():
            param.requires_grad = True

        for m in self.i3d.modules():
            if isinstance(m, nn.BatchNorm3d):
                m.weight.requires_grad = False
                m.bias.requires_grad = False

    def _image_to_head(self, pretrained='Kinetics'):
        if pretrained == 'ImageNet':
            net_conv = self.i3d.extract_features(self._image)
        elif pretrained == 'Kinetics':
            net_conv = self.i3d(self._image)
        return net_conv[:, :, 7, :, :], net_conv

    def load_pretrained_cnn(self, pretrained='Kinetics'):
        if pretrained == 'ImageNet':
            self.i3d.load_state_dict(torch.load('./weight/rgb_imagenet.pt'))
        elif pretrained == 'Kinetics':
            self.i3d.load_state_dict(torch.load('./weight/model_rgb.pth'))

    def create_architecture(self, num_classes, tag=None, anchor_scales=(8, 16, 32), anchor_ratios=(0.5, 1, 2), pretrained='Kinetics'):
        self._tag = tag

        self._num_classes = num_classes
        self._anchor_scales = anchor_scales
        self._num_scales = len(anchor_scales)

        self._anchor_ratios = anchor_ratios
        self._num_ratios = len(anchor_ratios)

        self._num_anchors = self._num_scales * self._num_ratios

        assert tag != None

        # Initialize layers
        self._init_modules(pretrained)

    def _init_modules(self, pretrained='Kinetics'):
        self._init_head_tail(pretrained)

        # rpn
        self.rpn_net = nn.Conv2d(cfg.RPN_CHANNELS, 512, kernel_size=3, padding=1)
        # self.linear_projection_3d = nn.Conv3d(cfg.RPN_CHANNELS, 128, kernel_size=(1, 1, 1))
        self.linear_projection_2d = nn.Conv2d(cfg.RPN_CHANNELS, 128, kernel_size=1)
        self.up_scale_2d = nn.Conv2d(128, cfg.RPN_CHANNELS, kernel_size=1)

        self.rpn_cls_score_net = nn.Conv2d(512, self._num_anchors * 2, kernel_size=1)
        self.rpn_bbox_pred_net = nn.Conv2d(512, self._num_anchors * 4, kernel_size=1)

        self.cls_score_net = nn.Linear(self._net_conv_channels, self._num_classes)
        self.bbox_pred_net = nn.Linear(self._net_conv_channels, self._num_classes * 4)

        self.transformer_block1 = TransformerBlock(self._net_conv_channels)
        self.transformer_block2 = TransformerBlock(self._net_conv_channels)
        self.transformer_block3 = TransformerBlock(self._net_conv_channels)

        self.init_weights()

    def init_weights(self):
        def normal_init(m, mean, stddev, truncated=False):
            if truncated:
                m.weight.data.normal_().fmod_(2).mul_(stddev).add_(mean)  # not a perfect approximation
            else:
                m.weight.data.normal_(mean, stddev)
            m.bias.data.zero_()

        normal_init(self.rpn_net, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.rpn_cls_score_net, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.rpn_bbox_pred_net, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.cls_score_net, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.bbox_pred_net, 0, 0.001, cfg.TRAIN.TRUNCATED)

    def _anchor_component(self, height, width):
        anchors, anchor_length = generate_anchors_pre(height, width, self._feat_stride, self._anchor_scales, self._anchor_ratios)
        self._anchors = torch.from_numpy(anchors).to(self._device)
        self._anchor_length = anchor_length

    def _proposal_layer(self, rpn_cls_prob, rpn_bbox_pred):
        rois, rpn_scores = proposal_layer(rpn_cls_prob, rpn_bbox_pred, self._im_info, self._mode, self._feat_stride, self._anchors, self._num_anchors)
        return rois, rpn_scores

    def _anchor_target_layer(self, rpn_cls_score):
        rpn_labels, rpn_bbox_targets, rpn_bbox_inside_weights, rpn_bbox_outside_weights = \
            anchor_target_layer(
                rpn_cls_score.data,
                self._gt_boxes.data.cpu().numpy(),
                self._im_info,
                self._feat_stride,
                self._anchors.data.cpu().numpy(),
                self._num_anchors,
            )

        rpn_labels = torch.from_numpy(rpn_labels).float().to(self._device)  #.set_shape([1, 1, None, None])
        rpn_bbox_targets = torch.from_numpy(rpn_bbox_targets).float().to(self._device)  #.set_shape([1, None, None, self._num_anchors * 4])
        rpn_bbox_inside_weights = torch.from_numpy(rpn_bbox_inside_weights).float().to(self._device)  #.set_shape([1, None, None, self._num_anchors * 4])
        rpn_bbox_outside_weights = torch.from_numpy(rpn_bbox_outside_weights).float().to(self._device)  #.set_shape([1, None, None, self._num_anchors * 4])

        rpn_labels = rpn_labels.long()
        self._anchor_targets['rpn_labels'] = rpn_labels
        self._anchor_targets['rpn_bbox_targets'] = rpn_bbox_targets
        self._anchor_targets['rpn_bbox_inside_weights'] = rpn_bbox_inside_weights
        self._anchor_targets['rpn_bbox_outside_weights'] = rpn_bbox_outside_weights

        return rpn_labels

    def _proposal_target_layer(self, rois, roi_scores):
        rois, roi_scores, labels, bbox_targets, bbox_inside_weights, bbox_outside_weights, cls_inside_weights = proposal_target_layer(rois, roi_scores, self._gt_boxes, self._gt_classes, self._num_classes)

        if rois is None:
            self._check = True

        if not self._check:
            self._proposal_targets['rois'] = rois
            # self._proposal_targets['labels'] = labels.long()
            self._proposal_targets['labels'] = labels
            self._proposal_targets['bbox_targets'] = bbox_targets
            self._proposal_targets['bbox_inside_weights'] = bbox_inside_weights
            self._proposal_targets['bbox_outside_weights'] = bbox_outside_weights
            self._proposal_targets['cls_inside_weights'] = cls_inside_weights

            return rois, roi_scores
        return None, None

    def _region_proposal(self, net_conv):
        rpn = F.relu(self.rpn_net(net_conv))

        rpn_cls_score = self.rpn_cls_score_net(rpn)  # batch * (num_anchors * 2) * h * w

        # change it so that the score has 2 as its channel size
        rpn_cls_score_reshape = rpn_cls_score.view(1, 2, -1, rpn_cls_score.size()[-1])  # batch * 2 * (num_anchors*h) * w
        rpn_cls_prob_reshape = F.softmax(rpn_cls_score_reshape, dim=1)

        # Move channel to the last dimension, to fit the input of python functions
        rpn_cls_prob = rpn_cls_prob_reshape.view_as(rpn_cls_score).permute(0, 2, 3, 1)  # batch * h * w * (num_anchors * 2)
        rpn_cls_score = rpn_cls_score.permute(0, 2, 3, 1)  # batch * h * w * (num_anchors * 2)
        rpn_cls_score_reshape = rpn_cls_score_reshape.permute(0, 2, 3, 1).contiguous()  # batch * (num_anchors*h) * w * 2
        rpn_cls_pred = torch.max(rpn_cls_score_reshape.view(-1, 2), 1)[1]

        rpn_bbox_pred = self.rpn_bbox_pred_net(rpn)
        rpn_bbox_pred = rpn_bbox_pred.permute(0, 2, 3, 1).contiguous()  # batch * h * w * (num_anchors*4)

        if self._mode == 'TRAIN':
            rois, roi_scores = self._proposal_layer(rpn_cls_prob, rpn_bbox_pred)  # rois, roi_scores are variable
            rpn_labels = self._anchor_target_layer(rpn_cls_score)
            rois, _ = self._proposal_target_layer(rois, roi_scores)
            if self._check:
                return None
        else:
            if cfg.TEST.MODE == 'nms':
                rois, _ = self._proposal_layer(rpn_cls_prob, rpn_bbox_pred)
            elif cfg.TEST.MODE == 'top':
                rois, _ = self._proposal_top_layer(rpn_cls_prob, rpn_bbox_pred)
            else:
                raise NotImplementedError

        self._predictions["rpn_cls_score"] = rpn_cls_score
        self._predictions["rpn_cls_score_reshape"] = rpn_cls_score_reshape
        self._predictions["rpn_cls_prob"] = rpn_cls_prob
        self._predictions["rpn_cls_pred"] = rpn_cls_pred
        self._predictions["rpn_bbox_pred"] = rpn_bbox_pred
        self._predictions["rois"] = rois

        return rois

    def _roi_pool_layer(self, bottom, rois):
        return RoIPool((cfg.POOLING_SIZE, cfg.POOLING_SIZE), 1.0 / 16.0)(bottom, rois)

    def _roi_align_layer(self, bottom, rois):
        return RoIAlign((cfg.POOLING_SIZE, cfg.POOLING_SIZE), 1.0 / 16.0, 0)(bottom, rois)

    def _query_preprocessing(self, query):
        return F.avg_pool2d(query, (7, 7))

    def _predict(self, pretrained='Kinetics'):
        torch.backends.cudnn.benchmark = False
        net_conv, video_feature = self._image_to_head(pretrained)

        # region proposal network, get rois
        self._anchor_component(net_conv.size(2), net_conv.size(3))
        rois = self._region_proposal(net_conv)
        if not self._check:
            if cfg.POOLING_MODE == 'align':
                pool5 = self._roi_align_layer(net_conv, rois)
            else:
                pool5 = self._roi_pool_layer(net_conv, rois)

            if self._mode == 'TRAIN':
                torch.backends.cudnn.benchmark = True
            avg_pool = F.avg_pool2d(pool5, (2, 2))
            avg_pool = self._query_preprocessing(avg_pool)
            down_sample_conv2d = self.linear_projection_2d(avg_pool)

            transformer_out1 = self.transformer_block1(down_sample_conv2d, video_feature)
            transformer_out2 = self.transformer_block2(transformer_out1, video_feature)
            transformer_out3 = self.transformer_block3(transformer_out2, video_feature)

            upscale_out = F.relu(self.up_scale_2d(transformer_out3))

            transformer_out3 = upscale_out.contiguous().view(-1, self._net_conv_channels)
            cls_prob, bbox_pred = self._region_classification(transformer_out3)

            return rois, cls_prob, bbox_pred
        return None, None, None

    def _region_classification(self, fc7):
        cls_score = self.cls_score_net(fc7)
        # cls_pred = torch.max(cls_score, 1)[1]
        cls_prob = torch.sigmoid(cls_score)
        bbox_pred = self.bbox_pred_net(fc7)

        self._predictions["cls_score"] = cls_score
        # self._predictions["cls_pred"] = cls_pred
        self._predictions["cls_prob"] = cls_prob
        self._predictions["bbox_pred"] = bbox_pred

        return cls_prob, bbox_pred

    def forward(self, image, im_info, boxes=None, gt_classes=None, mode='TRAIN'):
        self._image = image.to(self._device)
        self._im_info = im_info
        self._gt_boxes = boxes.to(self._device) if boxes is not None else None
        self._gt_classes = gt_classes.to(self._device) if gt_classes is not None else None

        self._mode = mode

        rois, cls_prob, bbox_pred = self._predict()
        if not self._check:
            if mode == 'TEST':
                stds = bbox_pred.data.new(cfg.TRAIN.BBOX_NORMALIZE_STDS).repeat(self._num_classes).unsqueeze(0).expand_as(bbox_pred)
                means = bbox_pred.data.new(cfg.TRAIN.BBOX_NORMALIZE_MEANS).repeat(self._num_classes).unsqueeze(0).expand_as(bbox_pred)
                self._predictions["bbox_pred"] = bbox_pred.mul(stds).add(means)
            else:
                self._add_losses()  # compute losses

        return rois

    def train_step(self, blobs, train_op):
        self._check = False
        rois = self.forward(blobs['data'], blobs['im_info'], blobs['boxes'], blobs['gt_classes'])
        if not self._check:
            rpn_loss_cls, rpn_loss_box, cross_entropy, loss_box, loss = self._losses["rpn_cross_entropy"].item(), \
                                                                        self._losses['rpn_loss_box'].item(), \
                                                                        self._losses['cross_entropy'].item(), \
                                                                        self._losses['loss_box'].item(), \
                                                                        self._losses['total_loss'].item()
            # rpn_loss_cls, rpn_loss_box, loss = self._losses["rpn_cross_entropy"].item(), \
            #                                   self._losses['rpn_loss_box'].item(), \
            #                                   self._losses['total_loss'].item()

            # utils.timer.timer.tic('backward')
            train_op.zero_grad()
            self._losses['total_loss'].backward()
            nn.utils.clip_grad_norm_(self.parameters(), max_norm=35, norm_type=2)
            # utils.timer.timer.toc('backward')
            train_op.step()

            self.delete_intermediate_states()

            return rpn_loss_cls, rpn_loss_box, cross_entropy, loss_box, loss
            # return rpn_loss_cls, rpn_loss_box, 0, 0, loss, rois
        return None, None, None, None, None

    def _add_losses(self, sigma_rpn=3.0):
        # RPN, class loss
        rpn_cls_score = self._predictions['rpn_cls_score_reshape'].view(-1, 2)
        rpn_label = self._anchor_targets['rpn_labels'].view(-1)
        rpn_select = (rpn_label.data != -1).nonzero().view(-1)

        rpn_cls_score = rpn_cls_score.index_select(0, rpn_select).contiguous().view(-1, 2)

        rpn_label = rpn_label.index_select(0, rpn_select).contiguous().view(-1)
        rpn_cross_entropy = F.cross_entropy(rpn_cls_score, rpn_label)

        # RPN, bbox loss
        rpn_bbox_pred = self._predictions['rpn_bbox_pred']
        rpn_bbox_targets = self._anchor_targets['rpn_bbox_targets']
        rpn_bbox_inside_weights = self._anchor_targets['rpn_bbox_inside_weights']
        rpn_bbox_outside_weights = self._anchor_targets['rpn_bbox_outside_weights']

        rpn_loss_box = self._smooth_l1_loss(
            rpn_bbox_pred,
            rpn_bbox_targets,
            rpn_bbox_inside_weights,
            rpn_bbox_outside_weights,
            sigma=sigma_rpn,
            dim=[1, 2, 3]
        )

        cls_score = self._predictions["cls_score"]
        # label = self._proposal_targets["labels"].view(-1)
        label = self._proposal_targets["labels"]
        # cross_entropy = F.cross_entropy(cls_score.view(-1, self._num_classes), label)
        cls_score = cls_score.view(-1, self._num_classes)
        cls_inside_weights = self._proposal_targets['cls_inside_weights']
        avg_factor = max(torch.sum(cls_inside_weights > 0).float().item(), 1.)
        cross_entropy = F.binary_cross_entropy_with_logits(cls_score, label.float(), weight=cls_inside_weights.float(), reduction='sum') / avg_factor

        # RCNN, bbox loss
        bbox_pred = self._predictions['bbox_pred']
        bbox_targets = self._proposal_targets['bbox_targets']
        bbox_inside_weights = self._proposal_targets['bbox_inside_weights']
        bbox_outside_weights = self._proposal_targets['bbox_outside_weights']
        # print(bbox_pred.size(), bbox_targets.size(), bbox_inside_weights.size(), bbox_outside_weights.size())
        loss_box = self._smooth_l1_loss(bbox_pred, bbox_targets, bbox_inside_weights, bbox_outside_weights)

        self._losses['cross_entropy'] = cross_entropy
        # self._losses['cross_entropy'] = 0
        self._losses['loss_box'] = loss_box
        # self._losses['loss_box'] = 0
        self._losses['rpn_cross_entropy'] = rpn_cross_entropy
        self._losses['rpn_loss_box'] = rpn_loss_box

        loss = cross_entropy + loss_box + rpn_cross_entropy + rpn_loss_box
        # loss = rpn_cross_entropy + rpn_loss_box
        self._losses['total_loss'] = loss

        # for k in self._losses.keys():
        #     self._event_summaries[k] = self._losses[k]

        return loss

    def _smooth_l1_loss(self,
                        bbox_pred,
                        bbox_targets,
                        bbox_inside_weights,
                        bbox_outside_weights,
                        sigma=1.0,
                        dim=[1]):
        sigma_2 = sigma**2
        box_diff = bbox_pred - bbox_targets
        in_box_diff = bbox_inside_weights * box_diff
        abs_in_box_diff = torch.abs(in_box_diff)
        smoothL1_sign = (abs_in_box_diff < 1. / sigma_2).detach().float()
        in_loss_box = torch.pow(in_box_diff, 2) * (sigma_2 / 2.) * smoothL1_sign + (abs_in_box_diff - (0.5 / sigma_2)) * (1. - smoothL1_sign)
        out_loss_box = bbox_outside_weights * in_loss_box
        loss_box = out_loss_box
        for i in sorted(dim, reverse=True):
            loss_box = loss_box.sum(i)
        loss_box = loss_box.mean()
        return loss_box

    def delete_intermediate_states(self):
        # Delete intermediate result to save memory
        for d in [
                self._losses, self._predictions, self._anchor_targets,
                self._proposal_targets
        ]:
            for k in list(d):
                del d[k]

    def test_image(self, image, im_info):
        self.eval()
        with torch.no_grad():
            self.forward(image, im_info, None, mode='TEST')
        cls_score, cls_prob, bbox_pred, rois = self._predictions["cls_score"].data.cpu().numpy(), \
                                               self._predictions["cls_prob"].data.cpu().numpy(), \
                                               self._predictions["bbox_pred"].data.cpu().numpy(), \
                                               self._predictions["rois"].data.cpu().numpy()

        return cls_score, cls_prob, bbox_pred, rois
