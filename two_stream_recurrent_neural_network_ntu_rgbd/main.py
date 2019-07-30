import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import os
import shutil
import time

from config import get_args
from ntu_rgb_preprocess import load_data
from model import Model


def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // 60))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def get_model_parameters(model):
    total_parameters = 0
    for layer in list(model.parameters()):
        layer_parameter = 1
        for l in list(layer.size()):
            layer_parameter *= l
        total_parameters += layer_parameter
    return total_parameters


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        print('Best Model Saving ...')
        shutil.copyfile(filename, 'model_best.pth.tar')


def train(model, train_loader, optimizer, criterion, epoch, args):
    model.train()

    train_acc = 0.0
    step = 0
    for x1, x2, x3, x4, x5, traversal, target in train_loader:
        start_time = time.time()
        adjust_learning_rate(optimizer, epoch, args)
        if args.cuda:
            x1, x2, x3, x4, x5, traversal, target =\
                x1.cuda(), x2.cuda(), x3.cuda(), x4.cuda(), x5.cuda(),\
                traversal.cuda(), target.cuda()

        optimizer.zero_grad()
        output = model(x1, x2, x3, x4, x5, traversal)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        y_pred = output.data.max(1)[1]
        acc = float(y_pred.eq(target.data).sum()) / len(x1) * 100.
        train_acc += acc
        step += 1
        if step % args.print_interval == 0:
            print("[Epoch {0:4d}] Loss: {1:2.3f} Acc: {2:.3f}%, time interval: {3}".format(epoch, loss.data, acc, time.time() - start_time), end=' ')
            for param_group in optimizer.param_groups:
                print(",  Current learning rate is: {}".format(param_group['lr']))


def eval(model, test_loader, args):
    print('evaluation ...')
    model.eval()
    correct = 0
    with torch.no_grad():
        for x1, x2, x3, x4, x5, traversal, target in test_loader:
            if args.cuda:
                x1, x2, x3, x4, x5, traversal, target = \
                    x1.cuda(), x2.cuda(), x3.cuda(), x4.cuda(), x5.cuda(), \
                    traversal.cuda(), target.cuda()
            output = model(x1, x2, x3, x4, x5, traversal)
            prediction = output.data.max(1)[1]
            correct += prediction.eq(target.data).sum()

    acc = 100. * float(correct) / len(test_loader.dataset)
    print('Test acc: {0:.2f}%'.format(acc))
    return acc


def main(args):
    # 1. load data
    train_loader, test_loader = load_data(args)

    # 2. define model
    model = Model(temporal_size=args.temporal_size, joint_sequence_size=117, num_classes=60)
    if args.pretrained:
        file_path = './model_best.pth.tar'
        checkpoint = torch.load(file_path)
        best_acc = checkpoint['best_acc1']
        model.load_state_dict(checkpoint['state_dict'])
        start_epoch = checkpoint['epoch']
        print('pretrained model loading ...')
        print('best acc : {}, start epoch : {}'.format(best_acc, start_epoch))
    else:
        best_acc = 0.0
        start_epoch = 1

    if args.cuda:
        model = model.cuda()
    print('model parameters :: {}'.format(get_model_parameters(model)))

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    for epoch in range(start_epoch, args.epochs + 1):
        train(model, train_loader, optimizer, criterion, epoch, args)
        eval_acc = eval(model, test_loader, args)

        is_best = eval_acc > best_acc
        best_acc = max(eval_acc, best_acc)
        save_checkpoint({
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'best_acc1': best_acc,
            'optimizer': optimizer.state_dict(),
            'parameters': get_model_parameters(model)
        }, is_best)


if __name__ == '__main__':
    args = get_args()
    main(args)
