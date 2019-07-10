import argparse


def get_args():
    parser = argparse.ArgumentParser('AVA dataset crawling, video crop, extract frame')

    parser.add_argument('--base-dir', type=str, default='D:/code_test', help='base directory we want to download')

    args = parser.parse_args()

    return args
