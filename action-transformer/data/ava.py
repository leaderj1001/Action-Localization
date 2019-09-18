import csv
import os
import pickle
import cv2


def read_csv(filename, data_mode='train'):
    with open(filename, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        return [data for data in reader]


class AVA(object):
    def __init__(self, image_set='train', base_dir='D:/'):
        # data mode
        self.image_set = image_set

        # path
        self.base_path = os.path.join(base_dir, 'AVA')
        self.annotation_path = os.path.join(self.base_path, 'annotation')
        self.frame_path = os.path.join(self.base_path, 'second_frame')

        # define class information
        self.classes = self._load_action_class()
        self.class_names = [value['name'] for value in self.classes.values()]
        self.num_classes = len(self.classes)
        self.class2idx = dict(list(zip(self.class_names, list(range(self.num_classes)))))

        # video information
        self.video_ext = '.mp4'
        self.frame_ext = '.jpg'
        self.video_index = self._load_video_set_index()
        self.fps = self._load_fps_dict()

        # get annotation ground truth
        self.roidb_handler = self._gt_roidb()

    def _load_action_class(self):
        """
        classnum : {
            name,
            label_type
        }
        """
        filename = os.path.join(self.annotation_path, 'ori_action_list.pkl')
        with open(filename, 'rb') as f:
            return pickle.load(f)

    def _load_video_set_index(self):
        """
        원하는 데이터 셋에 맞는 video name
        """
        video_set_file_path = os.path.join(self.base_path, 'annotation', 'Main', self.image_set + '.txt')
        with open(video_set_file_path) as f:
            video_index = [x.strip() for x in f.readlines()]
        return video_index

    def _load_fps_dict(self):
        filename = os.path.join(self.base_path, 'annotation', 'fps_dict.pkl')
        with open(filename, 'rb') as f:
            return pickle.load(f)

    def _load_ava_annotation(self):
        filename = os.path.join(self.base_path, 'annotation', 'ava_{}_v2.2.csv'.format(self.image_set))
        csv_data = read_csv(filename, data_mode=self.image_set)

        # 하나 더 추가
        csv_data.append(csv_data[-1])

        data_dict = {}
        count = 0

        gt_boxes = []
        gt_classes = []
        tmp = []
        for i, (line, next_line) in enumerate(zip(csv_data[:-1], csv_data[1:])):
            if line[0] not in self.video_index:
                continue

            video_name = line[0]
            time_stamp = line[1]

            boxes = [
                float(line[2]),
                float(line[3]),
                float(line[4]),
                float(line[5])
            ]
            action_id = int(line[6])
            person_id = int(line[7])

            fps = self.fps[video_name]
            frame_num = round(fps * (int(time_stamp) - 900))
            frame_range = list(range(frame_num - 32, frame_num + 32))
            image_name = ['{}_{}.jpg'.format(video_name, str(f_num).zfill(5)) for f_num in frame_range]
            image_path = [os.path.join(self.frame_path, video_name, i_name) for i_name in image_name]
            # print(fps, frame_num, image_name, image_path)
            # print(i, line)

            if line[2] == next_line[2] and line[3] == next_line[3] and line[4] == next_line[4] and line[5] == next_line[5]:
                tmp.append(action_id)
            else:
                gt_boxes.append(boxes)
                tmp.append(action_id)
                gt_classes.append(tmp)
                tmp = []

            if line[0] == next_line[0]:
                if line[1] != next_line[1]:
                    img = cv2.imread(image_path[32])
                    data_dict[count] = {
                        'video_name': video_name,
                        'time_stamp': time_stamp,
                        'image_name': image_name[32],
                        'image_path': image_path,
                        'image_height': img.shape[0],
                        'image_width': img.shape[1],
                        'boxes': gt_boxes,
                        'gt_classes': gt_classes,
                    }
                    gt_boxes = []
                    gt_classes = []
                    count += 1
            else:
                data_dict[count] = {
                    'video_name': video_name,
                    'time_stamp': time_stamp,
                    'image_name': image_name[32],
                    'image_path': image_path,
                    'image_height': img.shape[0],
                    'image_width': img.shape[1],
                    'boxes': gt_boxes,
                    'gt_classes': gt_classes,
                }
                gt_boxes = []
                gt_classes = []
                count += 1

            if i % 1000 == 0:
                # print(data_dict)
                print('current iteration :: {}'.format(i))

        return data_dict

    def _gt_roidb(self):
        init_annotation_cache_path = os.path.join(self.annotation_path, 'ava_{}_v2.2.pkl'.format(self.image_set))
        annotation_cache_path = os.path.join(self.annotation_path, 'transformer_ava_{}_v2.2.pkl'.format(self.image_set))

        if os.path.isfile(annotation_cache_path):
            print('file already exist !!')
            with open(os.path.join(self.annotation_path, 'transformer_ava_{}_v2.2.pkl').format(self.image_set), 'rb') as f:
                roidb = pickle.load(f)
            return roidb
        else:
            if os.path.isfile(init_annotation_cache_path):
                print('init annotation file already exist !!')
                with open(os.path.join(self.annotation_path, 'ava_{}_v2.2.pkl').format(self.image_set), 'rb') as f:
                    roidb = pickle.load(f)
            else:
                roidb = self._load_ava_annotation()

            # frame_data_dict = {}
            # dict_keys = list(roidb.keys())
            # cnt = 0
            # for idx in range(len(roidb)):
            #     if idx - 31 < 0:
            #         frame_range = dict_keys[0:idx + 33]
            #         key_frame = idx
            #     elif idx > 184346:
            #         frame_range = dict_keys[idx - 31:]
            #         key_frame = idx
            #     else:
            #         frame_range = dict_keys[idx - 31: idx + 33]
            #         key_frame = frame_range[31]
            #
            #     key_frame_dict = roidb[key_frame].copy()
            #     image_path = key_frame_dict['image_path']
            #     key_frame_dict['image_path'] = []
            #     key_frame_dict['image_path'].append(image_path)
            #     for frame_num in frame_range:
            #         line = roidb[frame_num]
            #         if frame_num == key_frame:
            #             continue
            #
            #         if key_frame_dict['video_name'] != line['video_name']:
            #             break
            #         key_frame_dict['image_path'].append(line['image_path'])
            #     frame_data_dict[cnt] = key_frame_dict
            #     cnt += 1

            with open(annotation_cache_path, 'wb') as f:
                pickle.dump(roidb, f, pickle.HIGHEST_PROTOCOL)
            print('save annotation !!')
            return roidb
