import torch
from torch.utils.data import Dataset, DataLoader

import numpy as np
import pickle
import math
import os


def load_pickle(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)


class DataSetLoader(Dataset):
    def __init__(self, filename, temporal_size=25):
        self.data = load_pickle(filename)
        self.len = len(self.data)
        self.temporal_size = temporal_size
        self.joint_size = 28

    def __getitem__(self, idx):
        data_hierarchical = torch.zeros((self.temporal_size, self.joint_size))
        data_traversal = torch.zeros((self.temporal_size, 70))
        values = self.data[idx]
        len_size = len(values)

        if len_size > self.temporal_size:
            lin = np.linspace(0, math.floor(len_size) - 1, self.temporal_size)
        else:
            lin = [j for j in range(len_size)]

        joint = []
        for l in lin:
            joint.append([float(i) for i in values[int(l)][5]])
        joint = torch.tensor(joint)

        data_hierarchical[:len_size, :] = joint

        data_transpose = data_hierarchical.t()
        data1_1 = data_transpose[4:10, :]
        data1_2 = data_transpose[10:16, :]
        data1_3 = torch.cat((data_transpose[0:4, :], (data_transpose[16:18, :] + data_transpose[18:20, :]) / 2), dim=0)
        data1_4 = data_transpose[16:22, :]
        data1_5 = data_transpose[22:, :]

        joint_dict = {}
        for i in range(14):
            joint_dict[i] = joint[:, i * 2:i * 2 + 2]
        traversal = [1, 2, 3, 4, 3, 2, 1, 5, 6, 7, 6, 5, 0, 1, 8, 9, 10, 9, 8, 11, 12, 13, 12, 11, 1]

        for i, each_traval in enumerate(traversal):
            data_traversal[:len_size, i * 2: i * 2 + 2] = joint_dict[each_traval]

        data2 = data_traversal
        label = torch.tensor(int(values[0][3]) - 1)

        return data1_1, data1_2, data1_3, data1_4, data1_5, data2, label

    def __len__(self):
        return self.len


def load_data(args):
    tubelet_annotation_path = os.path.join(args.base_dir, 'tubelet_annotation')
    tubelet_annotation_train = os.path.join(tubelet_annotation_path, 'tubelet_annotation_train.pkl')

    train_dataset = DataSetLoader(filename=tubelet_annotation_train, temporal_size=args.temporal_size)

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers
    )

    # tubelet_annotation_val = os.path.join(tubelet_annotation_path, 'tubelet_annotation_val.pkl')
    #
    # val_dataset = DataSetLoader(filename=tubelet_annotation_val, temporal_size=args.temporal_size)
    #
    # val_loader = DataLoader(
    #     dataset=val_dataset,
    #     batch_size=args.batch_size,
    #     shuffle=False,
    #     num_workers=args.num_workers
    # )

    return train_loader


from config import get_args
args = get_args()
train_loader = load_data(args)

for x1, x2, x3, x4, x5, xx, target in train_loader:
    break