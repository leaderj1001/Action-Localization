import glob
import cv2
import os
import math

from video_crawler import read_filenames
from config import get_args


def video2frame(args):
    if not os.path.isdir(args.video_path):
        os.mkdir(args.video_path)
    if not os.path.isdir(args.video_path + '/train'):
        os.mkdir(args.video_path + '/train')
    if not os.path.isdir(args.video_path + '/val'):
        os.mkdir(args.video_path + '/val')

    train_dir = os.path.join(args.video_path, 'train')
    val_dir = os.path.join(args.video_path, 'val')

    train_filenames = read_filenames(train_dir, mode='mp4')
    val_filenames = read_filenames(val_dir, mode='mp4')

    if not os.path.isdir(args.frame_path):
        os.mkdir(args.frame_path)
    if not os.path.isdir(args.frame_path + '/train'):
        os.mkdir(args.frame_path + '/train')
    if not os.path.isdir(args.frame_path + '/val'):
        os.mkdir(args.frame_path + '/val')

    train_frame_dir = os.path.join(args.frame_path, 'train')
    val_frame_dir = os.path.join(args.frame_path, 'val')

    train_file_list = []
    for train_filename in train_filenames:
        train_file_list.append([train_filename, os.path.basename(train_filename).split('.')[0]])

    for video_id in train_file_list:
        dir_name = os.path.join(train_frame_dir, video_id[1])
        if not os.path.isdir(dir_name):
            os.mkdir(dir_name)

        video_capture = cv2.VideoCapture(video_id[0])
        fps = math.ceil(video_capture.get(cv2.CAP_PROP_FPS))
        start_frame = 902
        end_frame = 1798

        count = 0
        while True:
            success, image = video_capture.read()
            if not success:
                print('Error')
                break

            num = int(count / fps)
            if start_frame <= num and num <= end_frame:
                frame = "{}.jpg".format("{0:05d}".format(num))
                image_name = os.path.join(dir_name, frame)
                cv2.imwrite(image_name, image)

            if cv2.waitKey(1) == fps:
                break
            count += 1

    # val video2frame
    val_file_list = []
    for val_filename in val_filenames:
        val_file_list.append([val_filename, os.path.basename(val_filename).split('.')[0]])

    for video_id in val_file_list:
        dir_name = os.path.join(val_frame_dir, video_id[1])
        if not os.path.isdir(dir_name):
            os.mkdir(dir_name)

        video_capture = cv2.VideoCapture(video_id[0])
        fps = math.ceil(video_capture.get(cv2.CAP_PROP_FPS))
        start_frame = 902
        end_frame = 1798

        count = 0
        while True:
            success, image = video_capture.read()
            if not success:
                print('Error')
                break

            num = int(count / fps)
            if start_frame <= num and num <= end_frame:
                frame = "{}.jpg".format("{0:05d}".format(num))
                image_name = os.path.join(dir_name, frame)
                cv2.imwrite(image_name, image)

            if cv2.waitKey(1) == fps:
                break
            count += 1


args = get_args()
video2frame(args)