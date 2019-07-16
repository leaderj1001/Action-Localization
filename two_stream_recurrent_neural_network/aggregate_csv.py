import glob
import os
import csv


def load_csv(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        data = [line for line in reader]

    return data


def aggregate_csv(args, data_mode='train'):
    joint_csv_path = os.path.join(args.base_dir, 'joint_csv/{}/*'.format(data_mode))
    joint_csv_list = glob.glob(joint_csv_path)
    print(joint_csv_path)
    print(len(joint_csv_list))

    out_dir = os.path.join(args.base_dir, 'aggregate_joint_{}.csv'.format(data_mode))
    print(out_dir)
    result = []
    for i, joint_csv in enumerate(joint_csv_list):
        print(joint_csv)
        data = load_csv(joint_csv)
        result.extend(data)
        # Empty list
        if not data:
            continue

    with open(out_dir, 'w', encoding='utf-8', newline='') as f:
        wr = csv.writer(f)
        for data in result:
            wr.writerow(data)
