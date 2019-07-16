import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import os
import cv2
import time
import glob
import matplotlib.pylab as plt
import numpy as np
import csv

from vgg19 import get_model
from post import decode_pose
import im_transform
from config import get_args


def vgg_preprocess(image):
    image = image.astype(np.float32) / 255.
    means = [0.485, 0.456, 0.406]
    stds = [0.229, 0.224, 0.225]

    preprocessed_img = image.copy()[:, :, ::-1]
    for i in range(3):
        preprocessed_img[:, :, i] = preprocessed_img[:, :, i] - means[i]
        preprocessed_img[:, :, i] = preprocessed_img[:, :, i] / stds[i]

    preprocessed_img = preprocessed_img.transpose((2, 0, 1)).astype(np.float32)
    return preprocessed_img


def rtpose_preprocess(image):
    image = image.astype(np.float32)
    image = image / 256. - 0.5
    image = image.transpose((2, 0, 1)).astype(np.float32)

    return image


def get_multiplier(img, version='light'):
    """Computes the sizes of image at different scales
    :param img: numpy array, the current image
    :returns : list of float. The computed scales
    """
    if version == 'light':
        scale_search = [1.]
    else:
        scale_search = [0.5, 1., 1.5, 2, 2.5]
    return [x * 368. / float(img.shape[0]) for x in scale_search]


def get_outputs(multiplier, img, model, preprocess):
    """Computes the averaged heatmap and paf for the given image
    :param multiplier:
    :param origImg: numpy array, the image being processed
    :param model: pytorch model
    :returns: numpy arrays, the averaged paf and heatmap
    """

    heatmap_avg = np.zeros((img.shape[0], img.shape[1], 19))
    paf_avg = np.zeros((img.shape[0], img.shape[1], 38))
    max_scale = multiplier[-1]
    max_size = max_scale * img.shape[0]
    # padding
    max_cropped, _, _ = im_transform.crop_with_factor(img, max_size, factor=8, is_ceil=True)
    batch_images = np.zeros((len(multiplier), 3, max_cropped.shape[0], max_cropped.shape[1]))

    for m in range(len(multiplier)):
        scale = multiplier[m]
        inp_size = scale * img.shape[0]

        # padding
        im_croped, im_scale, real_shape = im_transform.crop_with_factor(img, inp_size, factor=8, is_ceil=True)

        if preprocess == 'rtpose':
            im_data = rtpose_preprocess(im_croped)

        elif preprocess == 'vgg':
            im_data = vgg_preprocess(im_croped)

        batch_images[m, :, :im_data.shape[1], :im_data.shape[2]] = im_data

    # several scales as a batch
    batch_var = torch.from_numpy(batch_images).cuda().float()
    predicted_outputs, _ = model(batch_var)
    output1, output2 = predicted_outputs[-2], predicted_outputs[-1]
    heatmaps = output2.cpu().data.numpy().transpose(0, 2, 3, 1)
    pafs = output1.cpu().data.numpy().transpose(0, 2, 3, 1)

    for m in range(len(multiplier)):
        scale = multiplier[m]
        inp_size = scale * img.shape[0]

        # padding
        im_cropped, im_scale, real_shape = im_transform.crop_with_factor(img, inp_size, factor=8, is_ceil=True)
        heatmap = heatmaps[m, :int(im_cropped.shape[0] / 8), :int(im_cropped.shape[1] / 8), :]
        heatmap = cv2.resize(heatmap, None, fx=8, fy=8, interpolation=cv2.INTER_CUBIC)
        heatmap = heatmap[0:real_shape[0], 0:real_shape[1], :]
        heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_CUBIC)

        paf = pafs[m, :int(im_cropped.shape[0] / 8), :int(im_cropped.shape[1] / 8), :]
        paf = cv2.resize(paf, None, fx=8, fy=8, interpolation=cv2.INTER_CUBIC)
        paf = paf[0:real_shape[0], 0:real_shape[1], :]
        paf = cv2.resize(
            paf, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_CUBIC)

        heatmap_avg = heatmap_avg + heatmap / len(multiplier)
        paf_avg = paf_avg + paf / len(multiplier)

    return paf_avg, heatmap_avg


def handle_paf_and_heat(normal_heat, flipped_heat, normal_paf, flipped_paf):
    """Compute the average of normal and flipped heatmap and paf
    :param normal_heat: numpy array, the normal heatmap
    :param normal_paf: numpy array, the normal paf
    :param flipped_heat: numpy array, the flipped heatmap
    :param flipped_paf: numpy array, the flipped  paf
    :returns: numpy arrays, the averaged paf and heatmap
    """

    # The order to swap left and right of heatmap
    swap_heat = np.array((0, 1, 5, 6, 7, 2, 3, 4, 11, 12, 13, 8, 9, 10, 15, 14, 17, 16, 18))

    # paf's order
    # 0,1 2,3 4,5
    # neck to right_hip, right_hip to right_knee, right_knee to right_ankle

    # 6,7 8,9, 10,11
    # neck to left_hip, left_hip to left_knee, left_knee to left_ankle

    # 12,13 14,15, 16,17, 18, 19
    # neck to right_shoulder, right_shoulder to right_elbow, right_elbow to right_wrist, right_shoulder to right_ear

    # 20,21 22,23, 24,25 26,27
    # neck to left_shoulder, left_shoulder to left_elbow, left_elbow to left_wrist, left_shoulder to left_ear

    # 28,29, 30,31, 32,33, 34,35 36,37
    # neck to nose, nose to right_eye, nose to left_eye, right_eye to right_ear, left_eye to left_ear So the swap of paf should be:

    swap_paf = np.array((6, 7, 8, 9, 10, 11, 0, 1, 2, 3, 4, 5, 20, 21, 22, 23,
                         24, 25, 26, 27, 12, 13, 14, 15, 16, 17, 18, 19, 28,
                         29, 32, 33, 30, 31, 36, 37, 34, 35))
    flipped_paf = flipped_paf[:, ::-1, :]

    # The pafs are unit vectors, The x will change direction after flipped. not easy to understand, you may try visualize it.
    flipped_paf[:, :, swap_paf[1::2]] = flipped_paf[:, :, swap_paf[1::2]]
    flipped_paf[:, :, swap_paf[::2]] = -flipped_paf[:, :, swap_paf[::2]]
    averaged_paf = (normal_paf + flipped_paf[:, :, swap_paf]) / 2.
    averaged_heatmap = (normal_heat + flipped_heat[:, ::-1, :][:, :, swap_heat]) / 2.

    return averaged_paf, averaged_heatmap


def load_csv(filename):
    csv_dict = {}
    with open(filename, 'r', encoding='utf-8') as f:
        lines = csv.reader(f)

        for i, line in enumerate(lines):
            video_name = line[0]
            time_step = line[1]
            bbox = [float(line[2]), float(line[3]), float(line[4]), float(line[5])]
            action_id = line[6]
            person_id = line[7]

            key = video_name + '_' + time_step
            if key in csv_dict.keys():
                csv_dict[key].append([video_name, time_step, bbox, action_id, person_id])
            else:
                csv_dict[key] = [[video_name, time_step, bbox, action_id, person_id]]

    return csv_dict


def print_line():
    print('--' * 80)


def point2bbox(candidate, subset, height, width):
    point_subset = []
    joint = []
    for i, sub in enumerate(subset):
        joint_x, joint_y, bbox_x, bbox_y = [], [], [], []
        for j in sub[:18]:
            if j == -1.0:
                x, y = 0.0, 0.0
            else:
                x, y = candidate[int(j), :][0] / width, candidate[int(j), :][1] / height
                bbox_x.append(x)
                bbox_y.append(y)
            joint_x.append(x)
            joint_y.append(y)

        point_subset.append([
            round(min(bbox_x) / width, 3), round(min(bbox_y) / height, 3),
            round(max(bbox_x) / width, 3), round(max(bbox_y) / height, 3)
        ])
        temp = np.zeros([len(joint_x), 2])
        temp[:, 0] = joint_x
        temp[:, 1] = joint_y
        joint.append(temp)
    return np.array(point_subset), np.array(joint)


def bbox_overlaps(boxes, query_boxes):
    """
    Parameters
    ----------
    boxes: (N, 4) ndarray or tensor or variable
    query_boxes: (K, 4) ndarray or tensor or variable
    Returns
    -------
    overlaps: (N, K) overlap between boxes and query_boxes
    """
    if isinstance(boxes, np.ndarray):
        boxes = torch.from_numpy(boxes)
        query_boxes = torch.from_numpy(query_boxes)
        out_fn = lambda x: x.numpy()  # If input is ndarray, turn the overlaps back to ndarray when return
    else:
        out_fn = lambda x: x

    box_areas = (boxes[:, 2] - boxes[:, 0] + 1) * (boxes[:, 3] - boxes[:, 1] + 1)
    query_areas = (query_boxes[:, 2] - query_boxes[:, 0] + 1) * (query_boxes[:, 3] - query_boxes[:, 1] + 1)

    iw = (torch.min(boxes[:, 2:3], query_boxes[:, 2:3].t()) - torch.max(boxes[:, 0:1], query_boxes[:, 0:1].t()) + 1).clamp(min=0)
    ih = (torch.min(boxes[:, 3:4], query_boxes[:, 3:4].t()) - torch.max(boxes[:, 1:2], query_boxes[:, 1:2].t()) + 1).clamp(min=0)
    ua = box_areas.view(-1, 1) + query_areas.view(1, -1) - iw * ih
    overlaps = iw * ih / ua
    return out_fn(overlaps)


def main(args):
    weight_path = os.path.join(args.base_dir, 'weight/pose_model.pth')
    version = args.version
    print(weight_path, version)

    model = get_model('vgg19')
    model.load_state_dict(torch.load(weight_path))
    model.cuda()
    model.float()
    model.eval()

    joint_csv_base_dir = os.path.join(args.base_dir, 'joint_csv')
    print('joint_csv_base_dir :: {}'.format(joint_csv_base_dir))
    if not os.path.isdir(joint_csv_base_dir):
        os.mkdir(joint_csv_base_dir)

    annotation_dir = os.path.join(args.base_dir, 'annotation')
    for data_mode in ['train', 'val']:
        annotation_filename = 'ava_{}_v2.2.csv'.format(data_mode)
        annotation_path = os.path.join(annotation_dir, annotation_filename)

        csv_dict = load_csv(annotation_path)
        print('annotation filename :: {}, csv length :: {}'.format(annotation_filename, len(csv_dict)))
        print_line()

        frame_dir = os.path.join(args.base_dir, 'frame/trainval')
        frame_list = glob.glob(frame_dir + '/*')
        print('frame_list :: {}, frame_list length :: {}'.format(frame_list, len(frame_list)))
        print_line()

        joint_draw_base_dir = os.path.join(args.base_dir, 'draw_joint_frame')
        if not os.path.isdir(joint_draw_base_dir):
            os.mkdir(joint_draw_base_dir)

        joint_csv_dir = os.path.join(joint_csv_base_dir, data_mode)
        if not os.path.isdir(joint_csv_dir):
            os.mkdir(joint_csv_dir)
        print('joint csv dir :: {}'.format(joint_csv_dir))

        for directory in frame_list:
            result = []
            img_list = glob.glob(directory + '/*.jpg')
            video_name = os.path.basename(directory)
            joint_out_dir = os.path.join(joint_draw_base_dir, video_name)
            print('img_list length :: {}, directory path :: {}, base directory :: {}, joint out dir :: {}'.format(len(img_list), directory, video_name, joint_out_dir))

            if not os.path.isdir(joint_out_dir):
                os.mkdir(joint_out_dir)

            annotation_csv_filename = '{}_joint_ava_{}_v2.2.csv'.format(video_name, data_mode)
            if os.path.isfile(os.path.join(joint_csv_dir, annotation_csv_filename)):
                print('{} is already exist !!'.format(annotation_csv_filename))
                continue

            start_time = time.time()
            for index, img_path in enumerate(img_list):
                img = cv2.imread(img_path)
                if version == 'light':
                    img = cv2.resize(img, dsize=(224, 224), interpolation=cv2.INTER_AREA)
                height, width = img.shape[0], img.shape[1]
                img_filename = os.path.basename(img_path)
                frame_temp_step = int(img_filename.split(video_name)[1].split('_')[1]) + 900
                frame_count = img_filename.split(video_name)[1].split('_')[2].split('.')[0]
                if frame_count != '00':
                    continue

                key = video_name + '_' + str(frame_temp_step).zfill(4)
                print('key :: {}, frame_step :: {}, frame_count :: {}'.format(key, frame_temp_step, frame_count))
                multiplier = get_multiplier(img, version=version)

                model_run_start_time = time.time()
                with torch.no_grad():
                    paf, heatmap = get_outputs(multiplier, img, model, 'rtpose')
                    if version != 'light':
                        swapped_img = img[:, ::-1, :]
                        flipped_paf, flipped_heat = get_outputs(multiplier, swapped_img, model, 'rtpose')
                        paf, heatmap = handle_paf_and_heat(heatmap, flipped_heat, paf, flipped_paf)

                param = {
                    'thre1': 0.1,
                    'thre2': 0.05,
                    'thre3': 0.5
                }
                canvas, to_plot, candidate, subset = decode_pose(img, param, heatmap, paf)
                cv2.imwrite(os.path.join(joint_out_dir, img_filename), to_plot)

                bbox, joint = point2bbox(candidate, subset, height, width)
                if len(bbox) == 0:
                    continue

                if key in csv_dict.keys():
                    print('find csv_dict key...')
                    gt = np.array([value[2] for value in csv_dict[key]])
                    overlaps = bbox_overlaps(gt, bbox)
                    argmax_overlaps = overlaps.argmax(axis=1)

                    for idx, value in enumerate(csv_dict[key]):
                        joint_temp = joint[argmax_overlaps[idx], :]
                        temp = [
                            value[0], value[1], value[2][0], value[2][1], value[2][2], value[2][3], value[3], value[4]
                        ]
                        for each_joint in joint_temp[:14]:
                            temp.extend(each_joint)
                        result.append(temp)
                print('Each frame runtime :: {}'.format(time.time() - model_run_start_time))

            print('Each csv length :: ', len(result), 'Each runtime :: ', time.time() - start_time)
            with open(os.path.join(joint_csv_dir, annotation_csv_filename), 'w', encoding='utf-8', newline='') as f:
                wr = csv.writer(f)
                for data in result:
                    wr.writerow(data)


if __name__ == '__main__':
    args = get_args()
    main(args)

