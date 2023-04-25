import sys
import pickle

import torch
import scipy.linalg as scilin

import models
from utils import *
from args_unbalance import parse_eval_args
from datasets_unbalance import make_dataset
import time
import math
import numpy as np
import os
from sympy import Symbol, solve, S


MNIST_TRAIN_SAMPLES =[5923, 6742, 5958, 6131, 5842, 5421, 5918, 6265, 5851, 594]
MNIST_TEST_SAMPLES = [980, 1135, 1032, 1010, 982, 892, 958, 1028, 974, 1009]
CIFAR10_TRAIN_SAMPLES = [100, 100, 200, 200, 300, 300, 400, 400, 500, 500]

CIFAR10_TEST_SAMPLES = 10 * (1000,)
NUM_TRAIN_SAMPLES = sum(CIFAR10_TRAIN_SAMPLES)


def compute_s(weight_decay, feature_decay_rate, depth, device):
    c = 1
    s_array = torch.zeros(len(CIFAR10_TRAIN_SAMPLES), device=device)
    x = compute_x(CIFAR10_TRAIN_SAMPLES, depth, weight_decay, feature_decay_rate, device=device)
    s_array = pow(NUM_TRAIN_SAMPLES*feature_decay_rate *
                  pow(x, depth) / c, 1/(2*depth))
    
    return s_array

def compute_x(NUM_CLASS_SAMPLES: list, M: float, lambda_weight: float, lambda_feature: float, device):
    N = sum(NUM_CLASS_SAMPLES)
    b = M * N * math.pow(N * math.pow(lambda_weight, M) * lambda_feature, 1/M)
    x = Symbol('x', real=True)
    result = torch.empty(len(NUM_CLASS_SAMPLES), device=device)
    for i in range(len(NUM_CLASS_SAMPLES)):
        result[i] = torch.from_numpy(np.array(max(solve(b / NUM_CLASS_SAMPLES[i] - (M * x**(M-1))/((x ** M + 1)**2), x, rational=False)), dtype=np.float32)).to(device)
        if result[i] < 0:
            result[i] = 0
    return result

def compute_NC2_W_GOF(W, s_array, depth):
    K = W.shape[0]
    WWT = torch.mm(W, W.T)
    WWT /= torch.norm(WWT, p='fro')
    diag_array = torch.pow(s_array, 2* depth)
    sub = torch.diag_embed(diag_array)
    sub /= torch.norm(sub, p='fro')
    metric = torch.norm(WWT - sub, p='fro')
    return metric.detach().cpu().numpy().item()

def compute_NC2_H_GOF(mu_c_dict, s_array, feature_decay_rate, depth):
    K = len(mu_c_dict)
    H = torch.empty(mu_c_dict[0].shape[0], K, device = s_array.device)
    for i in range(K):
        H[:, i] = mu_c_dict[i]
    HTH = torch.mm(H.T, H)
    HTH /= torch.norm(HTH, p='fro')
    s_array_pow_M = pow(s_array, 2*depth)
    diag_array = torch.div(s_array_pow_M, 
                           pow((s_array_pow_M + NUM_TRAIN_SAMPLES * feature_decay_rate), 2))
    sub = torch.diag_embed(diag_array)
    sub /= torch.norm(sub, p='fro')
    metric = torch.norm(HTH - sub, p='fro')
    return metric.detach().cpu().numpy().item()

def compute_NC3_WH(W, mu_c_dict, s_array, feature_decay_rate, depth):
    K = len(mu_c_dict)
    H = torch.empty(mu_c_dict[0].shape[0], K, device = W.device)
    for i in range(K):
        H[:, i] = mu_c_dict[i]
    s_array_square_M = pow(s_array, 2*depth)
    diag_array = torch.div(s_array_square_M, 
                           s_array_square_M + NUM_TRAIN_SAMPLES * feature_decay_rate)
    WH = torch.mm(W, H.cuda())
    WH /= torch.norm(WH, p='fro')
    sub = torch.diag_embed(diag_array)
    sub /= torch.norm(sub, p='fro')
    res = torch.norm(WH - sub, p='fro')
    return res.detach().cpu().numpy().item()

def compute_NC3_W_sub_H(W, mu_c_dict, s_array, feature_decay_rate, depth):
    K = len(mu_c_dict)
    H = torch.empty(mu_c_dict[0].shape[0], K, device = W.device)
    diag_array = torch.zeros(len(CIFAR10_TRAIN_SAMPLES), device = W.device)
    for i in range(K):
        H[:, i] = mu_c_dict[i]
        diag_array[i] = s_array[i]**(2*depth) + NUM_TRAIN_SAMPLES*feature_decay_rate
    W /= torch.norm(W, p='fro')
    sub = torch.diag_embed(diag_array) @ H.T
    sub /= torch.norm(sub, p='fro')
    res = torch.norm(W - sub, p='fro')
    return res.detach().cpu().numpy().item()

def compute_NC3_W_div_H(W, mu_c_dict, s_array,feature_decay_rate, CIFAR10_TRAIN_SAMPLES, depth):
    K = len(mu_c_dict)
    H = torch.empty(mu_c_dict[0].shape[0], K, device = W.device)
    for i in range(K):
        H[:, i] = mu_c_dict[i]
    HTH = torch.mm(H.T, H)
    WWT = torch.mm(W, W.T)
    if depth == 1:
        diag = torch.FloatTensor(CIFAR10_TRAIN_SAMPLES).to(W.device)
    else:
        diag = pow((pow(s_array, 2*depth) + NUM_TRAIN_SAMPLES * feature_decay_rate), 2)
    res = torch.norm(torch.diagonal(WWT) / torch.diagonal(HTH) - diag)
    return res
    
class FCFeatures:
    def __init__(self):
        self.outputs = []

    def __call__(self, module, module_in):
        self.outputs.append(module_in)

    def clear(self):
        self.outputs = []


def compute_info(args, model, fc_features, dataloader, isTrain=True):
    mu_G_list = [0] * (len(model.fc))
    pairs = [(i, 0) for i in range(len(CIFAR10_TRAIN_SAMPLES))]
    mu_c_dict_list = [dict(pairs) for i in range(len(model.fc))]
    mu_c_dict_list_count = [dict(pairs) for i in range(len(model.fc))]
    top1 = AverageMeter()
    top5 = AverageMeter()
    
    for batch_idx, (inputs, targets) in enumerate(dataloader):

        inputs, targets = inputs.to(args.device), targets.to(args.device)

        with torch.no_grad():
            outputs = model(inputs)
        
        prec1, prec5 = compute_accuracy(outputs[0].data, targets.data, topk=(1, 5))
        top1.update(prec1.item(), inputs.size(0))
        top5.update(prec5.item(), inputs.size(0))
        if isTrain is True:
            features_list = []
            for i in range(len(model.fc)):
                features_list.append(fc_features[i].outputs[0][0])
                fc_features[i].clear()
                mu_G_list[i] += torch.sum(features_list[i], dim=0)
            for i in range(len(model.fc)):
                for y in range(len(CIFAR10_TRAIN_SAMPLES)):
                    indexes = (targets == y).nonzero(as_tuple=True)[0]
                    if indexes.nelement()==0:
                        mu_c_dict_list[i][y] += 0
                    else:
                        mu_c_dict_list[i][y] += features_list[i][indexes, :].sum(dim=0)
                        mu_c_dict_list_count[i][y] += indexes.shape[0]
    if args.dataset == 'mnist':
        pass
    elif args.dataset == 'cifar10' or args.dataset == 'cifar10_random':
        if isTrain:
            
            for i in range(len(model.fc)):
                mu_G_list[i] /= sum(CIFAR10_TRAIN_SAMPLES)
                for j in range(len(CIFAR10_TRAIN_SAMPLES)):
                    mu_c_dict_list[i][j] /= CIFAR10_TRAIN_SAMPLES[j]
        else:
            pass
    if isTrain:
        return mu_G_list, mu_c_dict_list, top1.avg, top5.avg
    else:
        return top1.avg, top5.avg


def compute_Sigma_W(args, model, fc_features, mu_c_dict, dataloader, isTrain=True):

    Sigma_W = 0
    for batch_idx, (inputs, targets) in enumerate(dataloader):

        inputs, targets = inputs.to(args.device), targets.to(args.device)

        with torch.no_grad():
            outputs = model(inputs)

        features = fc_features.outputs[0][0]
        fc_features.clear()
        debug_list = []
        # for b in range(len(targets)):
        #     y = targets[b].item()
        #     if y == 0:
        #         debug_list.append(features[b, :])
        #     Sigma_W += (features[b, :] - mu_c_dict[y]).unsqueeze(1) @ (features[b, :] - mu_c_dict[y]).unsqueeze(0)
        for y in range(len(mu_c_dict)):
            indexes = (targets == y).nonzero(as_tuple=True)[0]
            if indexes.nelement()==0:
                pass
            else:
                Sigma_W += ((features[indexes, :] - mu_c_dict[y]).unsqueeze(2) @ (features[indexes, :] - mu_c_dict[y]).unsqueeze(1)).sum(0)

    if args.dataset == 'mnist':
        if isTrain:
            Sigma_W /= sum(MNIST_TRAIN_SAMPLES)
        else:
            Sigma_W /= sum(MNIST_TEST_SAMPLES)
    elif args.dataset == 'cifar10' or args.dataset == 'cifar10_random':
        if isTrain:
            Sigma_W /= sum(CIFAR10_TRAIN_SAMPLES)
        else:
            Sigma_W /= sum(CIFAR10_TEST_SAMPLES)

    return Sigma_W.cpu().numpy()


def compute_Sigma_B(mu_c_dict, mu_G):
    Sigma_B = 0
    K = len(mu_c_dict)
    for i in range(K):
        Sigma_B += (mu_c_dict[i] - mu_G).unsqueeze(1) @ (mu_c_dict[i] - mu_G).unsqueeze(0)

    Sigma_B /= K

    return Sigma_B.cpu().numpy()


def main():
    args = parse_eval_args()
    args.batch_size = sum(CIFAR10_TRAIN_SAMPLES)
    name = "unbalanced_" + args.dataset + "-" + args.model \
           + "-" + args.loss + "-" + args.optimizer \
           + "-width_" + str(args.width) \
           + "-depth_relu_" + str(args.depth_relu) \
           + "-depth_linear_" + str(args.depth_linear) \
           + "-bias_" + str(args.bias) + "-" + "lr_" + str(args.lr) \
           + "-data_augmentation_" + str(args.data_augmentation) \
           + "-" + "epochs_" + str(args.epochs) + "-" + "seed_" + str(args.seed)
    load_path = os.path.join("model_weights", name)
    name = "eval-" + name



    device = torch.device("cuda:" + str(args.gpu_id) if torch.cuda.is_available() else "cpu")
    args.device = device
    args.batch_size = sum(CIFAR10_TRAIN_SAMPLES)
    set_seed(manualSeed = args.seed)

    if args.dataset == "cifar10":
        trainloader, testloader, num_classes = make_dataset(args.dataset, args.data_dir,
                                                            CIFAR10_TRAIN_SAMPLES, args.batch_size,
                                                            args.sample_size)
    
    if args.model == "MLP":
        model = models.__dict__[args.model](hidden = args.width, depth_relu = args.depth_relu, depth_linear = args.depth_linear, fc_bias=args.bias, num_classes=num_classes, batchnorm=False).to(device)
    elif args.model.startswith("VGG"):
        model = models.__dict__[args.model](hidden = args.width, depth_linear = args.depth_linear, pretrained=False, num_classes=num_classes, fc_bias=args.bias).to(device)
    elif args.model.startswith("ResNet"):
        model = models.__dict__[args.model](hidden = args.width, depth_linear = args.depth_linear, pretrained=False, num_classes=num_classes, fc_bias=args.bias).to(device)
    elif args.model.startswith("shufflenet"):
        model = models.__dict__[args.model](hidden = args.width, depth_linear = args.depth_linear, pretrained=False, num_classes=num_classes, fc_bias=args.bias).to(device)

    fc_features = [FCFeatures() for i in range(len(model.fc))]

    for i in reversed(range(len(model.fc))):
        model.fc[i].register_forward_pre_hook(fc_features[len(model.fc) - 1 - i])

    s_array = compute_s(args.weight_decay, args.feature_decay_rate, args.depth_linear, device=device)

    for epoch in range(args.epochs):
        if epoch % 100 == 0 or epoch == args.epochs - 1:
            pass
        else:
            continue
        not_available = True
        while not_available is True:
            not_available = False
            try:
                model.load_state_dict(torch.load(load_path + "/" + 'epoch_' + str(epoch + 1).zfill(3) + '.pth'))
                not_available = False
            except Exception as e:
                not_available = True
                print("Waiting for load when model available")
                print(load_path + "/" + 'epoch_' + str(epoch + 1).zfill(3) + '.pth')
                time.sleep(60)

        model.eval()
        start = time.time()
        W_list = []
        W_temp = model.fc[-1].weight.clone()
        W_list.append(W_temp)
        for j in list(reversed(range(0, len(model.fc)-1))):
            if isinstance(model.fc[j], nn.Linear):
                W_temp = W_temp @ model.fc[j].weight
                W_list.append(W_temp)
        if args.bias is True:
            b = model.fc[-1].bias

        mu_G_train_dict, mu_c_dict_train_dict, train_acc1, train_acc5 = compute_info(args, model, fc_features, trainloader, isTrain=True)

        Sigma_W = compute_Sigma_W(args, model, fc_features[-1], mu_c_dict_train_dict[-1], trainloader, isTrain=True)
        Sigma_B = compute_Sigma_B(mu_c_dict_train_dict[-1], mu_G_train_dict[-1])

        NC1_collapse_metric = np.trace(Sigma_W @ scilin.pinv(Sigma_B)) / len(mu_c_dict_train_dict[-1])

        NC2_W = compute_NC2_W_GOF(W_list[-1].clone(), s_array, args.depth_linear)
        NC2_H = compute_NC2_H_GOF(mu_c_dict_train_dict[-1], s_array, args.feature_decay_rate, args.depth_linear)
        NC3_W_sub_H = compute_NC3_W_sub_H(W_list[-1].clone(), mu_c_dict_train_dict[-1], s_array,
                                          args.feature_decay_rate, args.depth_linear)
        NC3_WH = compute_NC3_WH(W_list[-1].clone(), mu_c_dict_train_dict[-1],
                                s_array, args.feature_decay_rate, args.depth_linear)
        NC3_W_div_H = compute_NC3_W_div_H(W_list[-1].clone(), mu_c_dict_train_dict[-1], s_array, args.feature_decay_rate, CIFAR10_TRAIN_SAMPLES, args.depth_linear)


        print(time.time()-start)



if __name__ == "__main__":
    main()