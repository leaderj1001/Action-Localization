# Two Stream Recurrent Neural Network, NTU RGB+D Dataset

## Usage
1. Follow the link below to perform human skeleton data preprocessing.
  - [Link](https://github.com/leaderj1001/Action-Localization/tree/master/two_stream_recurrent_neural_network_ntu_rgbd/ntu-dataset)
2. Run the main.py.
  ```python
  python main.py
  ```

## Data Tree
```bash
├── two_stream_recurrent_neural_network_ntu_rgbd
    ├── ntu-dataset
    |   ├── README.md
    |   ├── ntu_dataset_main.py
    |   ├── all_train_sample.pkl
    |   └── all_test_sample.pkl
    ├── config.py
    ├── main.py
    ├── model.py
    └── ntu_rgb_preprocess.py
```
- Executing ntu_dataset_main.py will generate 'all_train_sample.pkl' and 'all_test_sample.pkl' files.
