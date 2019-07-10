# Atomic Visual Actions (AVA) Dataset Crawler

## Data Tree (initial setting)
```bash
├── D:/
    ├── data
        ├── annotation
        |   ├── ava_train_v2.2.csv
        |   └── ava_val_v2.2.csv
        ├── ava_file_names
        |   ├── ava_file_names_test_v2.1.txt
            └── ava_file_names_trainval_v2.1.txt
```
  - Annotation Download ([link](https://research.google.com/ava/download.html))
  - ava_file_names Download ([link](https://github.com/leaderj1001/Action-Localization/issues/1))
- v2.1 and v2.2 video dataset is same. So, we can use v2.1.txt

## Usage
```python
python video_crawler.py
```
- you have to set your own base_dir.<br><br>

Options:
- `--base_dir` (str) - (default: 'D:/code_test')

## Data Tree (after execute video_crawler.py)
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
    └── frame
        ├── trainval
        └── test
```

## Requirements
- urllib==3.7
- ffmpeg
  - [install link for Window](http://blog.gregzaal.com/how-to-install-ffmpeg-on-windows/)
