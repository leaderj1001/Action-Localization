import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torchvision.ops import nms

import numpy as np
import os
import csv

from network import Network
from config import get_args
from utils.config import cfg
from preprocess import load_data
from nets.bbox_transform import clip_boxes, bbox_transform_inv, bbox_overlaps

os.environ["CUDA_VISIBLE_DEVICES"] = "1"


def _clip_boxes(boxes, im_shape):
    """Clip boxes to image boundaries."""
    # x1 >= 0
    boxes[:, 0::4] = np.maximum(boxes[:, 0::4], 0)
    # y1 >= 0
    boxes[:, 1::4] = np.maximum(boxes[:, 1::4], 0)
    # x2 < im_shape[1]
    boxes[:, 2::4] = np.minimum(boxes[:, 2::4], im_shape[1] - 1)
    # y2 < im_shape[0]
    boxes[:, 3::4] = np.minimum(boxes[:, 3::4], im_shape[0] - 1)
    return boxes


def construct_graph(model):
    model.create_architecture(num_classes=args.num_classes + 1,
                              tag='default',
                              anchor_scales=cfg.ANCHOR_SCALES,
                              anchor_ratios=cfg.ANCHOR_RATIOS)

    lr = cfg.TRAIN.LEARNING_RATE
    params = []
    for key, value in dict(model.named_parameters()).items():
        if value.requires_grad:
            if 'bias' in key:
                params += [{
                    'params': [value],
                    'lr': lr * (cfg.TRAIN.DOUBLE_BIAS + 1),
                    'weight_decay': cfg.TRAIN.BIAS_DECAY and cfg.TRAIN.WEIGHT_DECAY or 0
                }]
            else:
                params += [{
                    'params': [value],
                    'lr': lr,
                    'weight_decay': getattr(value, 'weight_decay', cfg.TRAIN.WEIGHT_DECAY)
                }]
    optimizer = optim.SGD(model.parameters(), lr=cfg.TRAIN.LEARNING_RATE, momentum=cfg.TRAIN.MOMENTUM, weight_decay=cfg.TRAIN.WEIGHT_DECAY)

    return optimizer


def train(model, train_loader, epoch, args):
    optimizer = construct_graph(model)

    if args.use_pretrained:
        filename = 'E:/action_transformer_checkpoints3/faster_rcnn_action_epoch_1_iter_2900.pth'
        print('load pretrained model ... {}'.format(filename))
        model.load_state_dict(torch.load(filename))
    else:
        if args.use_backbone_pretrained == 'ImageNet':
            print('load backbone pretrained model ... {}'.format('ImageNet'))
            model.load_pretrained_cnn(pretrained='ImageNet')
        elif args.use_backbone_pretrained == 'Kinetics':
            print('load backbone pretrained model ... {}'.format('Kinetics'))
            model.load_pretrained_cnn(pretrained='Kinetics')

    model.train()
    model.cuda()

    small_bbox = []
    for idx, blobs in enumerate(train_loader):
        # print(blobs['video_name'], blobs['timestamps'])
        blobs['boxes'] = blobs['boxes'].squeeze(dim=0).view(-1, 4)
        blobs['gt_classes'] = blobs['gt_classes'].squeeze(dim=0).view(-1, 81)
        # print(blobs['gt_classes'].size())
        blobs['im_info'] = np.array(blobs['im_info'].squeeze(dim=0))
        # print(blobs['boxes'])
        # print(blobs['gt_classes'])

        rpn_loss_cls, rpn_loss_box, cross_entropy, loss_box, loss = model.train_step(blobs, train_op=optimizer)
        # overlaps = bbox_overlaps(rois[:, 1:], blobs['boxes'].cuda())

        if idx % args.print_interval == 0:
            filename = cfg.TRAIN.SNAPSHOT_PREFIX + '_epoch_{}'.format(epoch) + '_iter_{:d}'.format(idx) + '.pth'
            filename = os.path.join('E:/action_transformer_checkpoints3', filename)
            torch.save(model.state_dict(), filename)
            print('save model ... {}'.format(filename))

        if rpn_loss_cls is not None:
            print('[Epoch :: {0:4d}], rpn_loss_cls :: {1:.4f}, rpn_loss_box :: {2:.4f}, cross_entropy :: {3:.4f}, loss_box :: {4:.4f}, loss :: {5:.4f}'.format(
                epoch, rpn_loss_cls, rpn_loss_box, cross_entropy, loss_box, loss))
        else:
            small_bbox.append(idx)
            print(small_bbox)


def im_detect(model, data, im_info):
    _, scores, bbox_pred, rois = model.test_image(data, im_info)

    boxes = rois[:, 1:5]
    scores = np.reshape(scores, [scores.shape[0], -1])
    bbox_pred = np.reshape(bbox_pred, [bbox_pred.shape[0], -1])

    if cfg.TEST.BBOX_REG:
        box_deltas = bbox_pred
        pred_boxes = bbox_transform_inv(torch.from_numpy(boxes), torch.from_numpy(box_deltas)).numpy()
        pred_boxes = _clip_boxes(pred_boxes, im_shape=[400, 400, 3])
    else:
        pred_boxes = np.tile(boxes, (1, scores.shape[1]))

    return scores, pred_boxes


def _eval(model, test_loader, args, max_per_image=100, thresh=0.):
    optimizer = construct_graph(model)

    model.cuda()

    all_boxes = [[[] for _ in range(len(test_loader.dataset))] for _ in range(args.num_classes + 1)]

    if args.use_pretrained:
        filename = 'E:/action_transformer_checkpoints3/faster_rcnn_action_epoch_1_iter_7100.pth'
        print('load pretrained model ... {}'.format(filename))
        model.load_state_dict(torch.load(filename))

    all_boxes_dict = {}
    with torch.no_grad():
        for i, blobs in enumerate(test_loader):
            blobs['im_info'] = np.array(blobs['im_info'].squeeze(dim=0))
            video_name = blobs['video_name'][0]
            timestamps = blobs['timestamps'][0]

            if video_name not in all_boxes_dict.keys():
                all_boxes_dict[video_name] = {}
            if timestamps not in all_boxes_dict[video_name].keys():
                all_boxes_dict[video_name][timestamps] = {}

            scores, boxes = im_detect(model, blobs['data'], blobs['im_info'])

            # for j in range(1, args.num_classes + 1):
            #     inds = np.where(scores[:, j] > 0.5)[0]
            #     print(scores[inds, j])
            #     break
            # print(scores.shape, boxes.shape)

            for j in range(1, args.num_classes + 1):
                inds = np.where(scores[:, j] > 0.5)[0]
                cls_scores = scores[inds, j]
                cls_boxes = boxes[inds, j * 4:(j + 1) * 4]
                cls_dets = np.hstack((cls_boxes, cls_scores[:, np.newaxis])).astype(np.float32, copy=False)
                keep = nms(torch.from_numpy(cls_boxes), torch.from_numpy(cls_scores), cfg.TEST.NMS).numpy() if cls_dets.size > 0 else []
                cls_dets = cls_dets[keep, :]
                all_boxes[j][i] = cls_dets

            # Limit to max_per_image detections *over all classes*
            if max_per_image > 0:
                image_scores = np.hstack([all_boxes[j][i][:, -1] for j in range(1, args.num_classes + 1)])
                if len(image_scores) > max_per_image:
                    image_thresh = np.sort(image_scores)[-max_per_image]
                    for j in range(1, args.num_classes + 1):
                        keep = np.where(all_boxes[j][i][:, -1] >= image_thresh)[0]
                        if all_boxes[j][i][keep, :].size != 0:
                            # print(all_boxes[j][i][keep, :])
                            all_boxes_dict[video_name][timestamps][j] = all_boxes[j][i][keep, :]
                        all_boxes[j][i] = all_boxes[j][i][keep, :]
                else:
                    if len(image_scores) != 0:
                        image_thresh = np.sort(image_scores)[-len(image_scores)]
                        for j in range(1, args.num_classes + 1):
                            keep = np.where(all_boxes[j][i][:, -1] >= image_thresh)[0]
                            if all_boxes[j][i][keep, :].size != 0:
                                all_boxes_dict[video_name][timestamps][j] = all_boxes[j][i][keep, :]
                            all_boxes[j][i] = all_boxes[j][i][keep, :]
            print('[{0} / {1}] infer ...'.format(i, len(test_loader.dataset)))

            if i % args.print_interval == 0:
                print('Saving results ...')
                det_file = os.path.join('evaluation', 'results.csv')
                with open(det_file, 'w', encoding='utf-8', newline='') as f:
                    wr = csv.writer(f)
                    for m_name in all_boxes_dict.keys():
                        for time in all_boxes_dict[m_name].keys():
                            for class_info in all_boxes_dict[m_name][time]:
                                for box in all_boxes_dict[m_name][time][class_info]:
                                    # print(m_name, time, box[0], box[1], box[2], box[3], class_info, box[4])
                                    # wr.writerow([m_name, time, box[0] / 400.0, box[1] / 400.0, box[2] / 400.0, box[3] / 400.0, class_info, box[4]])
                                    wr.writerow([m_name, time, str(round(box[0] / 400.0, 3)), str(round(box[1] / 400.0, 3)),
                                         str(round(box[2] / 400.0, 3)), str(round(box[3] / 400.0, 3)), class_info, str(round(box[4], 3))])


def main(args):
    model = Network()

    train_loader, test_loader = load_data(args)

    if args.flag:
        _eval(model, test_loader, args)
    else:
        for epoch in range(1, args.epochs + 1):
            train(model, train_loader, epoch, args)
        _eval(model, test_loader, args)


if __name__ == '__main__':
    args = get_args()
    main(args)
