import csv
import os
import pickle


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
            joints = [
                line[8], line[9], line[10], line[11], line[12],
                line[13], line[14], line[15], line[16], line[17],
                line[18], line[19], line[20], line[21], line[22],
                line[23], line[24], line[25], line[26], line[27],
                line[28], line[29], line[30], line[31], line[32],
                line[33], line[34], line[35]
            ]

            keys.append(video_name)

            if video_name in name_dict.keys():
                name_dict[video_name].extend([[video_name, time_step, bbox, action_id, person_id, joints]])
            else:
                name_dict[video_name] = [[video_name, time_step, bbox, action_id, person_id, joints]]

        return name_dict, keys
    else:
        for i, name in enumerate(set(data)):
            for line in dictionary[name]:
                video_name = line[0]
                time_step = line[1]
                bbox = line[2]
                action_id = line[3]
                person_id = line[4]
                joints = line[5]

                if type == "action_id":
                    key = video_name + '_' + action_id
                elif type == 'person_id':
                    key = video_name + '_' + action_id + '_' + person_id

                keys.append(key)

                if key in name_dict.keys():
                    name_dict[key].extend([[video_name, time_step, bbox, action_id, person_id, joints]])
                else:
                    name_dict[key] = [[video_name, time_step, bbox, action_id, person_id, joints]]

        return name_dict, keys


def make_tubelet(data, dictionary):
    sample = {}
    count = 0
    for i, name in enumerate(set(data)):
        start_time = -1
        tubelet = []

        for line in dictionary[name]:
            if start_time == -1:
                tubelet.append([line[0], line[1], line[2], line[3], line[4], line[5]])
                start_time = line[1]
            else:
                # connected tubelet
                if int(line[1]) - int(start_time) == 1:
                    tubelet.append([line[0], line[1], line[2], line[3], line[4], line[5]])
                    start_time = line[1]
                else:
                    sample[count] = tubelet
                    tubelet = []
                    count += 1
                    tubelet.append([line[0], line[1], line[2], line[3], line[4], line[5]])
                    start_time = line[1]
        if len(tubelet) != 0:
            sample[count] = tubelet
            tubelet = []
            count += 1

    return sample


def csv2pickle(data, base_dir='', data_mode='train'):
    print('saving pickle ...')
    filename = os.path.join(base_dir, 'tubelet_annotation_{}.pkl'.format(data_mode))
    with open(filename, "wb") as f:
        pickle.dump(data, f)


def make_tubelet_main(args, data_mode):
    print('base_dir :: {}'.format(args.base_dir))
    annotation_path = os.path.join(args.base_dir, 'aggregate_joint_{}.csv'.format(data_mode))
    print('annotation path :: {}'.format(annotation_path))
    annotation_pkl_path = os.path.join(args.base_dir, 'tubelet_annotation')

    if not os.path.isdir(annotation_pkl_path):
        os.mkdir(annotation_pkl_path)

    data = read_csv(annotation_path)
    dictionary, keys = data_aggregation(data)
    dictionary, keys = data_aggregation(keys, dictionary, type='action_id')
    dictionary, keys = data_aggregation(keys, dictionary, type='person_id')

    data_sample = make_tubelet(keys, dictionary)
    print(len(data_sample))
    csv2pickle(data_sample, annotation_pkl_path)

    os.remove(annotation_path)
