# Action-Localization

## Dataset
- AVA(A Video Dataset of Atomic Visual Action) Challenge 2019
- [AVA dataset](https://research.google.com/ava/index.html)

## Video Crawling
```
pip install pytube
```
- pytube==9.5.0
```
python video_crawler.py --help
```
- `--data-path` : data path of ava.csv dataset. (default: ./data)
- `--video-path` : the directory where you want to save the video. (default: ./video)
