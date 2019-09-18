from torch.utils.data import Dataset, DataLoader
import torch

import os
import pickle
from PIL import Image
import csv
import cv2
import numpy as np
import numpy.random as npr

from utils.config import cfg  # network parameters
from config import get_args  # 전반적인 parameters
from data.blob import prep_im_for_blob, im_list_to_blob, _get_image_blob
from data.ava import AVA


def _to_one_hot(y, n_dims, dtype=torch.cuda.FloatTensor):
    result = torch.tensor([])
    for _y in y:
        # c_0 = torch.ones((n_dims, 1), dtype=torch.float32)
        # c_0[_y] = 0

        c_1 = torch.zeros((1, n_dims), dtype=torch.float32)
        c_1[:, _y] = 1

        # tmp = torch.cat((c_0, c_1), dim=1).unsqueeze(dim=0)
        # result = torch.cat((result, tmp), dim=0)
        result = torch.cat((result, c_1), dim=0)
    return result.type(dtype)


class DataSetLoader(Dataset):
    def __init__(self, image_set='train'):
        self.voc_dataset = AVA(image_set)
        self.len = len(self.voc_dataset.roidb_handler)
        self.roidb = self.voc_dataset.roidb_handler
        self.max_bbox_len = 42
        self._num_classes = 81

    def __getitem__(self, idx):
        roidb = [self.roidb[idx]]

        im_blob = _get_image_blob(roidb)

        _, height, width, channels = im_blob.shape
        im_blob = im_blob.reshape([channels, 64, height, width])
        blobs = {
            'data': im_blob,
            'video_name': roidb[0]['video_name'],
            'timestamps': roidb[0]['time_stamp']
        }

        assert len(roidb) == 1, "Single batch only"

        roidb[0]['gt_classes'] = np.array(roidb[0]['gt_classes'])
        roidb[0]['boxes'] = np.array(roidb[0]['boxes'])

        if cfg.TRAIN.USE_ALL_GT:
            gt_inds = np.where(roidb[0]['gt_classes'] != 0)[0]
        else:
            gt_inds = np.where(roidb[0]['gt_classes'] != 0 & np.all(roidb[0]['gt_overlaps'].toarray() > -1.0, axis=1))[0]

        # gt_boxes = np.empty((len(gt_inds), 4), dtype=np.float32)

        # gt_boxes[:, 0:4] = roidb[0]['boxes'][gt_inds, :] * 400.0
        # gt_boxes[:, 4] = roidb[0]['gt_classes'][gt_inds]

        blobs['boxes'] = np.array(roidb[0]['boxes'], dtype=np.float32) * 400.0
        blobs['gt_classes'] = _to_one_hot(roidb[0]['gt_classes'], n_dims=self._num_classes)
        blobs['im_info'] = np.array([im_blob.shape[2], im_blob.shape[3], 1.0], dtype=np.float32)
        return blobs

    def __len__(self):
        return self.len


def load_data(args):
    train_dataset = DataSetLoader(image_set='train')

    # mean, std = get_mean_and_std(train_dataset)

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )

    test_dataset = DataSetLoader(image_set='val')

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    return train_loader, test_loader


# tensor([77.0620, 75.3741, 63.2471]) tensor([50.3315, 47.9526, 46.2641])
# mean, std
def get_mean_and_std(dataset):
    '''Compute the mean and std value of dataset.'''
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0)
    mean = torch.zeros(3)
    std = torch.zeros(3)
    print('==> Computing mean and std..')
    for idx, inputs in enumerate(dataloader):
        for i in range(3):
            mean[i] += inputs['data'][:, i, 0, :, :].mean()
            std[i] += inputs['data'][:, i, 0, :, :].std()
        print(mean / float(idx + 1), std / float(idx + 1))
    mean.div_(len(dataset))
    std.div_(len(dataset))
    print('get mean and std :: mean :: {0:.4f} std :: {1:.4f'.format(mean, std))
    return mean, std


# def main(args):
#     train_loader, test_loader = load_data(args)
#     for blob in train_loader:
#         print(blob['data'].size())
#         print(blob['video_name'])
#         print(blob['timestamps'])
#         print(blob['boxes'].size())
#         print(blob['gt_classes'].size())
#         break
#
#
# if __name__ == '__main__':
#     from config import get_args
#     args = get_args()
#
#     main(args)
