# Extract Joint using OpenPose

## Data Tree (init)
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
    └── weight
        └── pose_model.pth
```

## Usage
```python
python extract_joint.py
```
- you have to set your own base_dir.<br><br>
- If you use the 'light' version, it takes 0.2 seconds to run in one frame, and 2 seconds if you use 'heavy' version. But 'heavy' version is a little more accurate.

Options:
- `--base_dir` (str) - (default: 'D:/code_test')
- `--version` (str) - (default: 'light')

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

## Reference
- [Realtime Multi-Person 2D Pose Estimation using Part Affinity Fields Paper](https://arxiv.org/abs/1611.08050)
- [OpenPose github](https://github.com/CMU-Perceptual-Computing-Lab/openpose)
- [Pytorch OpenPose github](https://github.com/tensorboy/pytorch_Realtime_Multi-Person_Pose_Estimation)
