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
        self.joint_size = 60

        self.left_arm_num = [8, 9, 11, 23]
        self.right_arm_num = [4, 5, 7, 21]
        self.torso_num = [3, 20, 1, 0]
        self.left_leg_num = [16, 17, 18, 19]
        self.right_leg_num = [12, 13, 14, 15]
        self.traversal = [
            20, 8, 9, 11, 23,
            11, 9, 8, 20, 4,
            5, 7, 21, 7, 5,
            4, 20, 3, 20, 1,
            0, 16, 17, 18, 19,
            18, 17, 16, 0, 12,
            13, 14, 15, 14, 13,
            12, 0, 1, 20
        ]

    def __getitem__(self, idx):
        padded_joint = torch.zeros((self.temporal_size, 25, 3))
        values = self.data[idx]
        len_size = len(values[0])

        if len_size > self.temporal_size:
            lin = np.linspace(0, math.floor(len_size) - 1, self.temporal_size)
        else:
            lin = [j for j in range(len_size)]
        all_joint = np.array(values[0])
        joint = []
        for l in lin:
            joint.append(all_joint[int(l), :, :])
        joint = torch.tensor(joint)

        padded_joint[:len_size, :] = joint

        center_x, center_y, center_z = (padded_joint[:, 16, 0] + padded_joint[:, 12, 0]) / 2, (padded_joint[:, 16, 1] + padded_joint[:, 12, 1]) / 2, (padded_joint[:, 16, 2] + padded_joint[:, 12, 2]) / 2
        padded_joint[:, :, 0] -= center_x.view(self.temporal_size, 1)
        padded_joint[:, :, 1] -= center_y.view(self.temporal_size, 1)
        padded_joint[:, :, 2] -= center_z.view(self.temporal_size, 1)

        label = int(values[1]) - 1

        left_arm = torch.cat([padded_joint[:, i, :] for i in self.left_arm_num], dim=1).t()
        right_arm = torch.cat([padded_joint[:, i, :] for i in self.right_arm_num], dim=1).t()
        torso = torch.cat([padded_joint[:, i, :] for i in self.torso_num], dim=1).t()
        left_leg = torch.cat([padded_joint[:, i, :] for i in self.left_leg_num], dim=1).t()
        right_leg = torch.cat([padded_joint[:, i, :] for i in self.right_leg_num], dim=1).t()

        traversal = torch.cat([padded_joint[:, i, :] for i in self.traversal], dim=1)

        return left_arm, right_arm, torso, left_leg, right_leg, traversal, label

    def __len__(self):
        return self.len


def load_data(args):
    train_annotation = './ntu-dataset/all_train_sample_each_person.pkl'
    test_annotation = './ntu-dataset/all_test_sample_each_person.pkl'

    train_dataset = DataSetLoader(filename=train_annotation, temporal_size=args.temporal_size)

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers
    )

    test_dataset = DataSetLoader(filename=test_annotation, temporal_size=args.temporal_size)

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers
    )

    return train_loader, test_loader
