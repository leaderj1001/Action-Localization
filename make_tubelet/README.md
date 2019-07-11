# Atomic Visual Actions (AVA) Dataset, Make Tubelet
- Use the ground truth to sample the tubelets that are related to each other.

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

## Example
### raw data
![init](https://user-images.githubusercontent.com/22078438/61024906-ff73d700-a3e9-11e9-80b6-4dab02e349d0.PNG)
- You can see that not all data is connected.

### Make Tubelet
![tubelet](https://user-images.githubusercontent.com/22078438/61024979-26caa400-a3ea-11e9-8bc0-fc2248e3ae59.PNG)
- Depending on the time, action_id, and person_id, you can see that the related ground truths have created a tubelet.

## Requirements
- csv
- pickle
