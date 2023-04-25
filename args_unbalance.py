import os
import shutil
import datetime
import argparse

import torch
import numpy as np


def parse_train_args():
    parser = argparse.ArgumentParser()

    # parameters
    # Model Selection
    parser.add_argument('--model', type=str, default='resnet18')
    parser.add_argument('--no-bias', dest='bias', action='store_false')
    parser.add_argument('--ETF_fc', dest='ETF_fc', action='store_true')
    parser.add_argument('--fixdim', dest='fixdim', type=int, default=0)
    parser.add_argument('--SOTA', dest='SOTA', action='store_true')
    parser.add_argument('--data_augmentation', dest='data_augmentation', action='store_true')
    
    # MLP settings (only when using mlp and res_adapt(in which case only width has effect))
    parser.add_argument('--width', type=int, default=1024)
    parser.add_argument('--depth_relu', type=int, default=6)

    # Hardware Setting
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--seed', type=int, default=6)
    parser.add_argument('--use_cudnn', type=bool, default=True)

    # Directory Setting
    parser.add_argument('--dataset', type=str, choices=['mnist', 'cifar10', 'cifar10_random', 'fashionmnist'], default='mnist')
    parser.add_argument('--data_dir', type=str, default='~/data')
    # parser.add_argument('--uid', type=str, default=None)
    parser.add_argument('--force', action='store_true', help='force to override the given uid')

    # Learning Options
    parser.add_argument('--epochs_pretrain', type=int, default=24, help='Max Epochs')
    parser.add_argument('--epochs', type=int, default=200, help='Max Epochs')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--loss', type=str, default='MSE', help='loss function configuration')
    parser.add_argument('--sample_size', type=int, default=None, help='sample size PER CLASS')

    # Optimization specifications
    parser.add_argument('--lr_pretrain', type=float, default=0.1, help='pretrain learning rate')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--patience_pretrain', type=int, default=6, help='learning rate decay per N epochs')
    parser.add_argument('--patience', type=int, default=100, help='learning rate decay per N epochs')
    parser.add_argument('--decay_type', type=str, default='step', help='learning rate decay type')
    parser.add_argument('--gamma', type=float, default=0.1, help='learning rate decay factor for step decay')
    parser.add_argument('--gamma_pretrain', type=float, default=0.1, help='learning rate decay factor for step decay')
    parser.add_argument('--optimizer', default='SGD', help='optimizer to use')
    parser.add_argument('--optimizer_pretrain', default='SGD', help='optimizer to use')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='weight decay')
    # The following two should be specified when testing adding wd on Features
    parser.add_argument('--sep_decay', action='store_true', help='whether to separate weight decay to last feature and last weights')
    parser.add_argument('--feature_decay_rate', type=float, default=1e-5, help='weight decay for last layer feature')
    parser.add_argument('--history_size', type=int, default=10, help='history size for LBFGS')
    parser.add_argument('--ghost_batch', type=int, dest='ghost_batch', default=128, help='ghost size for LBFGS variants')
    parser.add_argument('--pretrained', dest='pretrained', action='store_true')
    parser.add_argument('--depth_linear', type=int, default=1, dest='depth_linear')

    args = parser.parse_args()

    args.timestamp = datetime.datetime.now().strftime("%a-%b-%d-%H:%M:%S")
    
    name = "unbalanced_" + args.dataset + "-" + args.model \
           + "-" + args.loss + "-" + args.optimizer \
           + "-width_" + str(args.width) \
           + "-depth_relu_" + str(args.depth_relu) \
           + "-depth_linear_" + str(args.depth_linear) \
           + "-bias_" + str(args.bias) + "-" + "lr_" + str(args.lr) \
           + "-data_augmentation_" + str(args.data_augmentation) \
           + "-" + "epochs_" + str(args.epochs) + "-" + "seed_" + str(args.seed)
    save_path = './model_weights/' + name
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)

    parser.add_argument('--save_path', default=save_path, help='the output dir of weights')
    parser.add_argument('--log', default=save_path + '/log.txt', help='the log file in training')
    parser.add_argument('--arg', default=save_path + '/args.txt', help='the args used')
    parser.add_argument('--timestamp', default=args.timestamp)

    args = parser.parse_args()

    with open(args.log, 'w') as f:
        f.close()
    with open(args.arg, 'w') as f:
        print(args)
        print(args, file=f)
        f.close()
    if args.use_cudnn:
        print("cudnn is used")
        torch.backends.cudnn.benchmark = True
    else:
        print("cudnn is not used")
        torch.backends.cudnn.benchmark = False

    return args


def parse_eval_args():
    parser = argparse.ArgumentParser()

    # parameters
    # Model Selection
    parser.add_argument('--model', type=str, default='resnet18')
    parser.add_argument('--no-bias', dest='bias', action='store_false')
    parser.add_argument('--ETF_fc', dest='ETF_fc', action='store_true')
    parser.add_argument('--fixdim', dest='fixdim', type=int, default=0)
    parser.add_argument('--SOTA', dest='SOTA', action='store_true')
    parser.add_argument('--data_augmentation', dest='data_augmentation', action='store_true')
    
    # MLP settings (only when using mlp and res_adapt(in which case only width has effect))
    parser.add_argument('--width', type=int, default=1024)
    parser.add_argument('--depth_relu', type=int, default=6)

    # Hardware Setting
    parser.add_argument('--gpu_id', type=int, default=0)

    # Directory Setting
    parser.add_argument('--dataset', type=str, choices=['mnist', 'cifar10', 'cifar10_random'], default='mnist')
    parser.add_argument('--data_dir', type=str, default='~/data')

    # Learning Options
    parser.add_argument('--epochs', type=int, default=200, help='Max Epochs')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--sample_size', type=int, default=None, help='sample size PER CLASS')
    parser.add_argument('--depth_linear', type=int, default=1, dest='depth_linear')
    parser.add_argument('--seed', type=int, default=6)
    parser.add_argument('--loss', type=str, default='MSE', help='loss function configuration')
    parser.add_argument('--sep_decay', action='store_true', help='whether to separate weight decay to last feature and last weights')
    parser.add_argument('--pretrain_lr', type=float, default=0.1, help='pretrain learning rate')
    parser.add_argument('--lr', type=float, default=0.1, help='learning rate')
    parser.add_argument('--optimizer', default='SGD', help='optimizer to use')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='weight decay')
    parser.add_argument('--feature_decay_rate', type=float, default=1e-5, help='weight decay for last layer feature')
    parser.add_argument('--patience', type=int, default=100, help='learning rate decay per N epochs')



    args = parser.parse_args()

    return args
