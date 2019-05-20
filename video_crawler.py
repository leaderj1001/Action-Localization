import json
import glob
import os
import csv
import pickle
from pytube import YouTube

from config import get_args

args = get_args()


def read_data(filename, mode=None):
    with open(filename, "r", encoding='utf-8') as f:
        if mode is None:
            return f.readlines()
        elif mode == "json":
            return json.load(f)
        elif mode == "csv":
            return [line for line in csv.reader(f)]
    if mode == 'pkl':
        with open(filename, 'rb') as f:
            return pickle.load(f)


def read_filenames(path, mode='csv'):
    return glob.glob(path + '/*.' + mode)


def write_pickle(filename, data):
    with open(filename, "wb") as f:
        pickle.dump(data, f)


def csv2pickle():
    train_dir = os.path.join(args.data_path, 'train')
    val_dir = os.path.join(args.data_path, 'val')

    train_filenames = read_filenames(train_dir, mode='csv')
    val_filenames = read_filenames(val_dir, mode='csv')

    count = 0
    train_dict = {}
    for train_filename in train_filenames:
        file = os.path.splitext(train_filename)[0] + '.pkl'
        if not os.path.isfile(file):
            print("read train csv data ...")
            data = read_data(train_filename, mode='csv')
            for line in data:
                train_dict[count] = {
                    "video_id": line[0],
                    "frame_timestamp": line[1],
                    "person_box": [line[2], line[3], line[4], line[5]],
                    "action_id": line[6],
                    "person_id": line[7]
                }
                count += 1
            write_pickle(file, train_dict)
        else:
            print("load pickle data ...")

    count = 0
    val_dict = {}
    for val_filename in val_filenames:
        file = os.path.splitext(val_filename)[0] + '.pkl'
        if not os.path.isfile(file):
            print("read train csv data ...")
            data = read_data(val_filename, mode='csv')
            for line in data:
                val_dict[count] = {
                    "video_id": line[0],
                    "frame_timestamp": line[1],
                    "person_box": [line[2], line[3], line[4], line[5]],
                    "action_id": line[6],
                    "person_id": line[7]
                }
                count += 1
            write_pickle(file, val_dict)
        else:
            print("load pickle data ...")

    return train_dir, val_dir


def load_pickle_data():
    train_dir, val_dir = csv2pickle()
    train_filenames = read_filenames(train_dir, mode='pkl')
    val_filenames = read_filenames(val_dir, mode='pkl')

    train_dict = {}
    for train_filename in train_filenames:
        data = read_data(train_filename, mode="pkl")
        train_dict.update(data)

    val_dict = {}
    for val_filename in val_filenames:
        data = read_data(val_filename, mode="pkl")
        val_dict.update(data)

    return train_dict, val_dict


def read_current_video():
    train_dir = os.path.join(args.video_path, "train")
    val_dir = os.path.join(args.video_path, "val")

    if not os.path.isdir(train_dir):
        os.mkdir(train_dir)
    if not os.path.isdir(val_dir):
        os.mkdir(val_dir)

    train_video_names = read_filenames(train_dir, mode='mp4')
    val_video_names = read_filenames(val_dir, mode='mp4')

    current_train_video_list = []
    current_val_video_list = []

    for train_video_name in train_video_names:
        current_train_video_list.append(os.path.basename(train_video_name).split('.')[0])
    for val_video_name in val_video_names:
        current_val_video_list.append(os.path.basename(val_video_name).split('.')[0])

    return current_train_video_list, current_val_video_list, train_dir, val_dir


def video_crawler():
    train_dict, val_dict = load_pickle_data()

    train_video_list = []
    val_video_list = []

    for line in train_dict.values():
        if line['video_id'] not in train_video_list:
            train_video_list.append(line['video_id'])

    for line in val_dict.values():
        if line['video_id'] not in val_video_list:
            val_video_list.append(line['video_id'])

    current_train_list, current_val_list, train_video_dir, val_video_dir = read_current_video()

    for train_video in train_video_list:
        if train_video not in current_train_list:
            url = "https://www.youtube.com/watch?v="
            try:
                youtube = YouTube(url + train_video)
                youtube.streams.filter(progressive=True).order_by('resolution').desc().first().download(train_video_dir, filename=train_video)
            except:
                print("download fail: ", train_video)

    for val_video in val_video_list:
        if val_video not in current_val_list:
            url = "https://www.youtube.com/watch?v="
            try:
                youtube = YouTube(url + val_video)
                youtube.streams.filter(progressive=True).order_by('resolution').desc().first().download(val_video_dir, filename=val_video)
            except:
                print("download fail: ", val_video)


video_crawler()
