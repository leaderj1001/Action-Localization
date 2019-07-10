import os
import urllib.request
import subprocess
import glob
import cv2
import math
import numpy as np

from config import get_args


def load_txt(file_path, mode='trainval'):
    filename = 'ava_file_names_{}_v2.1.txt'.format(mode)
    filename = os.path.join(file_path, filename)
    with open(filename, 'r') as f:
        video_names = f.readlines()
    return video_names


def is_video(video_name, output_dir):
    filename = os.path.join(output_dir, video_name)
    return os.path.isfile(filename)


def video_crawler(video_name, mode='trainval', output_dir=''):
    url = 'https://s3.amazonaws.com/ava-dataset/{}/{}'.format(mode, video_name)

    if is_video(video_name, output_dir):
        print('Already exist video: {0}'.format(video_name))
    else:
        print('Download video: {0}'.format(video_name))
        output_dir = os.path.join(output_dir, video_name)
        urllib.request.urlretrieve(url, output_dir)


def ava_crawler(output_dir, args, mode='trainval'):
    file_path = os.path.join(args.base_dir, 'data/ava_file_names')
    video_names = load_txt(file_path, mode)
    output_dir = os.path.join(output_dir, mode)

    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    for i, video_name in enumerate(video_names):
        print('count: {0}, video_name: {1}'.format(i, video_name))
        video_crawler(video_name, mode=mode, output_dir=output_dir)


def video_crop(video_name, video_path, cropped_dir, mode='trainval'):
    start_time = 900
    end_time = 1800

    origin_video_filename = '{}/{}'.format(video_path, video_name)
    cropped_video_filename = '{}/{}.mp4'.format(cropped_dir, video_name.split('.')[0])

    status = False
    if not os.path.isfile(origin_video_filename):
        print('Video does not exist: {0}'.format(video_name))
    elif os.path.isfile(cropped_video_filename):
        print('Already exist cropped video: {0}'.format(video_name))
    else:
        command = [
            'ffmpeg',
            '-i', '"%s"' % origin_video_filename,
            '-ss', str(start_time),
            '-t', str(end_time - start_time),
            '-c:v', 'libx264', '-c:a', 'ac3',
            '-threads', '1',
            '-loglevel', 'panic',
            '"{}"'.format(cropped_video_filename)
        ]
        command = ' '.join(command)

        try:
            print("\tProcessing video: {}".format(video_name))
            output = subprocess.check_output(command, shell=True, stderr=subprocess.STDOUT)
        except subprocess.CalledProcessError as err:
            # print('status :: ', status, ', error print :: ', err.output.decode('euc-kr'))
            return status, err.output

        status = os.path.exists(cropped_video_filename)

    return status


def process_frame(frame, output_folder, video_id, frame_number, current_second, resize_min_size=400, jpg_quality=85):
    # Compute output dimensions
    height, width, _ = frame.shape
    ratio = float(height) / float(width)
    if ratio > 1.0:
        W = resize_min_size
        H = int(ratio * float(W))
    else:
        H = resize_min_size
        W = int(float(H) / ratio)

    # Resize frame
    resized_frame = cv2.resize(frame, (W, H))

    # Generate destination path
    frame_number = str(frame_number)
    current_second = '0' * (4 - len(str(current_second))) + str(current_second)
    frame_number = '0' * (2 - len(frame_number))+frame_number
    dst_filename = "{}_{}_{}.jpg".format(video_id, current_second, frame_number)
    dst_filename = os.path.join(output_folder, dst_filename)

    # Save frame
    cv2.imwrite(dst_filename, resized_frame, [cv2.IMWRITE_JPEG_QUALITY, jpg_quality])


def video2frame(video_path, output_folder, video_id, resize_min_size=400, fps=25):
    print('video_path :: ', video_path)
    video_capture = cv2.VideoCapture(video_path)
    video_fps = video_capture.get(cv2.CAP_PROP_FPS)
    frame_count = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_time_step = 1 / float(frame_count)
    print("FPS: {}".format(video_fps))
    print("frameCount: {}".format(frame_count))
    print("frame_time_step: {}".format(frame_time_step))

    current_second = 0
    if video_fps > 29:
        lin = np.linspace(0, math.floor(video_fps), fps)
    else:
        lin = np.linspace(0, math.floor(video_fps)-1, fps)

    while video_capture.isOpened():
        f = 0
        frame_number = 0
        total_frame = 0
        for i, elem in enumerate(lin):
            ret, frame = video_capture.read()
            total_frame += 1
            if ret:
                process_frame(frame, output_folder, video_id, frame_number, current_second, resize_min_size=resize_min_size)
                frame_number += 1

                if i != 0:
                    f = int(elem) - int(lin[i-1])
                else:
                    f = int(lin[i+1]) - int(elem)

                for _ in range(f-1):
                    ret, frame = video_capture.read()
                    total_frame += 1

                    if total_frame >= video_fps:
                        break
            else:
                break

            if total_frame >= video_fps:
                break

        ret, frame = video_capture.retrieve()
        if not ret:
            break

        current_second += 1


def main(args):
    # 1. download ava video
    save_video_dir = os.path.join(args.base_dir, 'video')
    if not os.path.isdir(save_video_dir):
        os.mkdir(save_video_dir)

    for data_mode in ['trainval', 'test']:
        ava_crawler(save_video_dir, args, mode=data_mode)

    # 2. Crop video
    cropped_dir = os.path.join(args.base_dir, 'cropped_video')
    if not os.path.isdir(cropped_dir):
        os.mkdir(cropped_dir)

    for data_mode in ['trainval', 'test']:
        video_path = save_video_dir
        video_path = os.path.join(video_path, data_mode)

        cropped_path = os.path.join(cropped_dir, data_mode)
        if not os.path.isdir(cropped_path):
            os.mkdir(cropped_path)
        video_list = glob.glob(video_path + '/*')

        for video in video_list:
            video_name = os.path.basename(video)
            video_crop(video_name, video_path, cropped_path, mode=data_mode)

    # 3. Video to Frame
    frame_dir = os.path.join(args.base_dir, 'frame')
    if not os.path.isdir(frame_dir):
        os.mkdir(frame_dir)

    for data_mode in ['trainval', 'test']:
        cropped_path = os.path.join(cropped_dir, data_mode)
        video_list = glob.glob(cropped_path + '/*.mp4')

        frame_path = os.path.join(frame_dir, data_mode)
        if not os.path.isdir(frame_path):
            os.mkdir(frame_path)

        for video in video_list:
            video_name = os.path.basename(video).split('.')[0]
            out_dir = os.path.join(frame_path, video_name)

            if not os.path.isdir(out_dir):
                os.mkdir(out_dir)
            video2frame(video, out_dir, video_name)


if __name__ == "__main__":
    args = get_args()
    main(args)
