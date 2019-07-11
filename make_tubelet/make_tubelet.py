import csv
import os
import pickle
import argparse


def get_args():
    parser = argparse.ArgumentParser('make tubelet using ground truth')

    parser.add_argument('--base-dir', type=str, default='D:/code_test', help='base directory we want to download')

    args = parser.parse_args()

    return args


def read_csv(filename):
    print('read_csv filename {}'.format(filename))
    with open(filename, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        data = [line for line in reader]

    return data


def data_aggregation(data, dictionary=None, type=""):
    keys = []
    name_dict = {}
    if dictionary is None:
        for i, line in enumerate(data):

            video_name = line[0]
            time_step = line[1]
            bbox = [line[2], line[3], line[4], line[5]]
            action_id = line[6]
            person_id = line[7]

            keys.append(video_name)

            if video_name in name_dict.keys():
                name_dict[video_name].extend([[video_name, time_step, bbox, action_id, person_id]])
            else:
                name_dict[video_name] = [[video_name, time_step, bbox, action_id, person_id]]

        return name_dict, keys
    else:
        for i, name in enumerate(set(data)):
            for line in dictionary[name]:
                video_name = line[0]
                time_step = line[1]
                bbox = line[2]
                action_id = line[3]
                person_id = line[4]

                if type == "action_id":
                    key = video_name + '_' + action_id
                elif type == 'person_id':
                    key = video_name + '_' + action_id + '_' + person_id

                keys.append(key)

                if key in name_dict.keys():
                    name_dict[key].extend([[video_name, time_step, bbox, action_id, person_id]])
                else:
                    name_dict[key] = [[video_name, time_step, bbox, action_id, person_id]]

        return name_dict, keys


def make_tubelet(data, dictionary):
    sample = {}
    count = 0
    for i, name in enumerate(set(data)):
        start_time = -1
        tubelet = []

        for line in dictionary[name]:
            if start_time == -1:
                tubelet.append([line[0], line[1], line[2], line[3], line[4]])
                start_time = line[1]
            else:
                # 연결된 tubelet임.
                if int(line[1]) - int(start_time) == 1:
                    tubelet.append([line[0], line[1], line[2], line[3], line[4]])
                    start_time = line[1]
                else:
                    sample[count] = tubelet
                    tubelet = []
                    count += 1
                    tubelet.append([line[0], line[1], line[2], line[3], line[4]])
                    start_time = line[1]
        if len(tubelet) != 0:
            sample[count] = tubelet
            tubelet = []
            count += 1

    return sample


def csv2pickle(data, data_mode, base_dir=''):
    print('saving pickle ...')
    filename = os.path.join(base_dir, data_mode + '_tubelet_annotation.pkl')
    with open(filename, "wb") as f:
        pickle.dump(data, f)


def main(args):
    print('base_dir :: {}'.format(args.base_dir))
    annotation_path = os.path.join(args.base_dir, 'data/annotation')
    print('annotation path :: {}'.format(annotation_path))
    annotation_pkl_path = os.path.join(args.base_dir, 'data/tubelet_annotation')

    if not os.path.isdir(annotation_pkl_path):
        os.mkdir(annotation_pkl_path)

    for data_mode in ['train', 'val']:
        csv_file_path = os.path.join(annotation_path, 'ava_{}_v2.2.csv'.format(data_mode))
        print('data mode :: {}, csv file path :: {}'.format(data_mode, csv_file_path))

        data = read_csv(csv_file_path)
        dictionary, keys = data_aggregation(data)
        dictionary, keys = data_aggregation(keys, dictionary, type='action_id')
        dictionary, keys = data_aggregation(keys, dictionary, type='person_id')

        data_sample = make_tubelet(keys, dictionary)
        csv2pickle(data_sample, data_mode, annotation_pkl_path)


if __name__ == '__main__':
    args = get_args()
    main(args)