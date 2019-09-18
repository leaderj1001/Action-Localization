import argparse


def get_args():
    parser = argparse.ArgumentParser('action transformer')

    parser.add_argument('--base-dir', type=str, default='D:/')

    parser.add_argument('--model', type=str, default='resnext')
    parser.add_argument('--model-depth', type=int, default=101)
    parser.add_argument('--pretrained-path', type=str, default='./weight/resnext-101-64f-kinetics.pth')
    parser.add_argument('--n-finetune-classes', type=int, default=100)
    parser.add_argument('--ft-begin-index', type=int, default=0)
    parser.add_argument('--num-classes', type=int, default=80)
    parser.add_argument('--resnet-shortcut', type=str, default='B')
    parser.add_argument('--sample-size', type=int, default=224)
    parser.add_argument('--sample-duration', type=int, default=16)
    parser.add_argument('--resnext_cardinality', default=32, type=int, help='ResNeXt cardinality')

    parser.add_argument('--cuda', type=bool, default=True)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--use_backbone_pretrained', type=str, default='Kinetics')
    parser.add_argument('--use_pretrained', type=bool, default=False)
    parser.add_argument('--print_interval', type=int, default=100)

    args = parser.parse_args()

    return args
