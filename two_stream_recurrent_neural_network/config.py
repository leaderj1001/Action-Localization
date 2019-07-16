import argparse


def get_args():
    parser = argparse.ArgumentParser('two stream recurrent neural network')

    parser.add_argument('--base-dir', type=str, default='D:/code_test')
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--num-workers', type=int, default=0)

    parser.add_argument('--cuda', type=bool, default=True)

    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--weight-decay', type=float, default=1e-5)
    parser.add_argument('--momentum', type=float, default=0.9)

    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--print-interval', type=int, default=100)

    parser.add_argument('--temporal-size', type=int, default=16)

    args = parser.parse_args()

    return args
