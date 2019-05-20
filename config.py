import argparse


def get_args():
    parser = argparse.ArgumentParser("parameters")

    parser.add_argument('--data-path', type=str, default='./data')
    parser.add_argument('--video-path', type=str, default='./video')

    return parser.parse_args()
