# Atomic Visual Actions (AVA) Dataset, Make Tubelet
- 

## Data Tree (initial setting)
```bash
├── D:/
    ├── data
        ├── annotation
        |   ├── ava_train_v2.2.csv
        |   └── ava_val_v2.2.csv
```
  - Annotation Download ([link](https://research.google.com/ava/download.html))

## Usage
```python
python make_tubelet.py
```
- you have to set your own base_dir.<br><br>

Options:
- `--base_dir` (str) - (default: 'D:/code_test')

## Data Tree (after execute video_crawler.py)
```bash
├── D:/
    ├── data
        ├── annotation
        |   ├── ava_train_v2.2.csv
        |   └── ava_val_v2.2.csv
        └── tubelet_annotation
            ├── train_tubelet_annotation.pkl
            └── val_tubelet_annotation.pkl
```

## Requirements
- csv
- pickle
