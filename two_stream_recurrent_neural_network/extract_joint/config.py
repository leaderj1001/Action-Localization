import argparse


def get_args():
    parser = argparse.ArgumentParser('extract joint information')

    parser.add_argument('--base-dir', type=str, default='D:/code_test')
    parser.add_argument('--version', type=str, default='light', help='you can choose, light or heavy')

    args = parser.parse_args()

    return args
