# Implementing Modeling Temporal Dynamics and Spatial Configurations of Actions Using Two-Stream Recurrent Neural Networks using Pytorch

## Method
![캡처](https://user-images.githubusercontent.com/22078438/61292733-f2515080-a80c-11e9-8dba-a4c21062bd83.PNG)

## Data Tree (after execute extract_joint.py)
```bash
├── D:/
    ├── data
    |   ├── annotation
    |   |   ├── ava_train_v2.2.csv
    |   |   └── ava_val_v2.2.csv
    |   └── ava_file_names
    |       ├── ava_file_names_test_v2.1.txt
    |       └── ava_file_names_trainval_v2.1.txt
    ├── video
    |   ├── trainval
    |   └── test
    ├── cropped_video
    |   ├── trainval
    |   └── test
    ├── frame
    |   ├── trainval
    |   └── test
    ├── weight
    |   └── pose_model.pth
    └── joint_csv
        ├── train
        └── val
```
- If you wonder video, cropped_video, frame directory, you can go [link](https://github.com/leaderj1001/Action-Localization/tree/master/video_crawler)
- If you wonder weight, joint_csv directory, you can go [link](https://github.com/leaderj1001/Action-Localization/tree/master/two_stream_recurrent_neural_network/extract_joint)

## Usage
```python
python main.py
```
- you have to set your own base_dir.<br><br>
- If you use the 'light' version, it takes 0.2 seconds to run in one frame, and 2 seconds if you use 'heavy' version. But 'heavy' version is a little more accurate.

Options:
- `--base_dir` (str) - (default: 'D:/code_test')
- `--batch-size` (int) - (default: 256)
- `--num-workers` (int) - (default: 2)

- `--cuda` (bool) - (default: True)

- `--lr` (float) - (default: 0.0001)
- `--weight-decay` (float) - (default: 1e-5)
- `--momentum` (float) - (default: 0.9)

- `--epochs` (int) - (defualt: 100)
- `--print-interval` (int) - (default: 100)

- `--temporal-size` (int) - (default: 16)

## Data Tree (after execute main.py)
```bash
├── D:/
    ├── data
    |   ├── annotation
    |   |   ├── ava_train_v2.2.csv
    |   |   └── ava_val_v2.2.csv
    |   └── ava_file_names
    |       ├── ava_file_names_test_v2.1.txt
    |       └── ava_file_names_trainval_v2.1.txt
    ├── video
    |   ├── trainval
    |   └── test
    ├── cropped_video
    |   ├── trainval
    |   └── test
    ├── frame
    |   ├── trainval
    |   └── test
    ├── weight
    |   └── pose_model.pth
    ├── joint_csv
    |   ├── train
    |   └── val
    └── tubelet_annotation
        ├── train
        └── val
```

## Reference
- [Modeling Temporal Dynamics and Spatial Configurations of Actions Using Two-Stream Recurrent Neural Networks Paper](https://arxiv.org/abs/1704.02581)
