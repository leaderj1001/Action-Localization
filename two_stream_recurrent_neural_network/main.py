import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import os
import shutil

from config import get_args
from aggregate_csv import aggregate_csv
from make_tubelet import make_tubelet_main
from preprocess import load_data
from model import Model


def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


def train(model, train_loader, optimizer, criterion, epoch, args):
    model.train()

    train_acc = 0.0
    step = 0
    for x1, x2, x3, x4, x5, xx, target in train_loader:
        # adjust_learning_rate(optimizer, epoch, args)
        if args.cuda:
            x1, x2, x3, x4, x5, xx, target = x1.cuda(), x2.cuda(), x3.cuda(), x4.cuda(), x5.cuda(), xx.cuda(), target.cuda()

        optimizer.zero_grad()
        output = model(x1, x2, x3, x4, x5, xx)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        y_pred = output.data.max(1)[1]

        acc = float(y_pred.eq(target.data).sum()) / args.batch_size * 100.
        train_acc += acc
        step += 1
        if step % args.print_interval == 0:
            print("[Epoch {0:4d}] Loss: {1:2.3f} Acc: {2:.3f}%".format(epoch, loss.data, acc))
            # for param_group in optimizer.param_groups:
            #     print(",  Current learning rate is: {}".format(param_group['lr']))


def eval(model, test_loader, args):
    print('evaluation ...')
    model.eval()
    pass
    correct = 0
    with torch.no_grad():
        for x1, x2, x3, x4, x5, xx, target in test_loader:
            if args.cuda:
                x1, x2, x3, x4, x5, xx, target = x1.cuda(), x2.cuda(), x3.cuda(), x4.cuda(), x5.cuda(), xx.cuda(), target.cuda()
            output = model(x1, x2, x3, x4, x5, xx)
            prediction = output.data.max(1)[1]
            correct += prediction.eq(target.data).sum()

    acc = 100. * float(correct) / len(test_loader.dataset)
    print('Test acc: {0:.2f}'.format(acc))
    return acc


def main(args):
    for data_mode in ['train']:
        if os.path.isfile(os.path.join(args.base_dir, 'tubelet_annotation/tubelet_annotation_{}.pkl'.format(data_mode))):
            print('Already exist pickle file !!')
            continue

        # 1. aggregate joint csv file
        aggregate_csv(args, data_mode)

        # 2. make tubelet using aggregate_joint.csv
        make_tubelet_main(args, data_mode)

    # 3. load data
    train_loader = load_data(args)

    # 4. define model
    model = Model(temporal_size=args.temporal_size)
    if args.cuda:
        model = model.cuda()
    start_epoch = 0

    criterion = nn.CrossEntropyLoss()
    # optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_acc = 0.0
    for epoch in range(start_epoch, args.epochs + 1):
        train(model, train_loader, optimizer, criterion, epoch, args)
        eval_acc = eval(model, args)

        is_best = eval_acc > best_acc
        best_acc = max(eval_acc, best_acc)
        save_checkpoint({
            'epoch': epoch,
            'arch': args.model_name,
            'state_dict': model.state_dict(),
            'best_acc1': best_acc,
            'optimizer': optimizer.state_dict(),
        }, is_best)


if __name__ == '__main__':
    args = get_args()
    main(args)
