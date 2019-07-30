import os
import csv
import pickle
import numpy as np
import argparse


def get_args():
    parser = argparse.ArgumentParser('NTU RGB+D Dataset Preprocessing')

    parser.add_argument('--file-path', type=str, default='D:/nturgb+d_skeletons')

    return parser.parse_args()


def csv2pickle(data, filename, base_dir='./', data_mode='train'):
    print('saving pickle ...')
    filename = os.path.join(base_dir, filename)
    with open(filename, "wb") as f:
        pickle.dump(data, f)


def load_data(filename):
    with open(filename, 'r') as f:
        return f.readlines()


def save_csv(result, out_dir):
    with open(out_dir, 'w', encoding='utf-8', newline='') as f:
        wr = csv.writer(f)
        for d in result:
            wr.writerow(d)


def main(args):
    skeleton_file_list = os.listdir(args.file_path)

    cross_val = [
        1, 2, 4, 5, 8,
        9, 13, 14, 15, 16,
        17, 18, 19, 25, 27,
        28, 31, 34, 35, 38
    ]

    train_sample = {}
    test_sample = {}
    train_sample_count = 0
    test_sample_count = 0
    for file in skeleton_file_list:
        filename = file.split('.')[0]
        subject = int(filename[9:12])
        label = int(filename[-3:])
        print(filename, label, subject)

        data = load_data(os.path.join(args.file_path, file))

        count = 1
        sub_sample = {}
        while len(data) != count:
            person_count = int(data[count])
            count += 1

            joint = np.zeros(shape=[25, 3], dtype=np.float)
            for index in range(person_count):
                joint_cnt = 2
                while joint_cnt != 27:
                    line = list(data[count + joint_cnt].strip().split(' '))
                    joint[joint_cnt - 2, :] = [float(n) for n in line[:3]]
                    joint_cnt += 1
                if index in sub_sample.keys():
                    sub_sample[index].append(joint)
                else:
                    sub_sample[index] = [joint]
                count += 27

        for key, item in sub_sample.items():
            if subject in cross_val:
                train_sample[train_sample_count] = [item, label]
                train_sample_count += 1
            else:
                test_sample[test_sample_count] = [item, label]
                test_sample_count += 1
            break

    csv2pickle(train_sample, filename='all_train_sample.pkl')
    csv2pickle(test_sample, filename='all_test_sample.pkl')


if __name__ == '__main__':
    args = get_args()
    main(args)
