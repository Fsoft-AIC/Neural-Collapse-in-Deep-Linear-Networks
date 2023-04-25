import sys
import pickle

import torch
import scipy.linalg as scilin

import models
from utils import *
from args import parse_eval_args
from datasets import make_dataset
import time


MNIST_TRAIN_SAMPLES = (5923, 6742, 5958, 6131, 5842, 5421, 5918, 6265, 5851, 5949)
MNIST_TEST_SAMPLES = (980, 1135, 1032, 1010, 982, 892, 958, 1028, 974, 1009)
CIFAR10_TRAIN_SAMPLES = 10 * (5000,)
CIFAR10_TEST_SAMPLES = 10 * (1000,)


class FCFeatures:
    def __init__(self):
        self.outputs = []

    def __call__(self, module, module_in):
        self.outputs.append(module_in)

    def clear(self):
        self.outputs = []


def compute_info(args, model, fc_features, dataloader, isTrain=True):
    mu_G_list = [0] * (len(model.fc))
    pairs = [(i, 0) for i in range(len(MNIST_TRAIN_SAMPLES))]
    mu_c_dict_list = [dict(pairs) for i in range(len(model.fc))]
    top1 = AverageMeter()
    top5 = AverageMeter()
    
    for batch_idx, (inputs, targets) in enumerate(dataloader):

        inputs, targets = inputs.to(args.device), targets.to(args.device)

        with torch.no_grad():
            outputs = model(inputs)

        features_list = []
        for i in range(len(model.fc)):
            features_list.append(fc_features[i].outputs[0][0])
            fc_features[i].clear()
        for i in range(len(model.fc)):
            mu_G_list[i] += torch.sum(features_list[i], dim=0)
        for i in range(len(model.fc)):
            for y in range(len(MNIST_TRAIN_SAMPLES)):
                indexes = (targets == y).nonzero(as_tuple=True)[0]
                if indexes.nelement()==0:
                    mu_c_dict_list[i][y] += 0
                else:
                    mu_c_dict_list[i][y] += features_list[i][indexes, :].sum(dim=0)

        prec1, prec5 = compute_accuracy(outputs[0].data, targets.data, topk=(1, 5))
        top1.update(prec1.item(), inputs.size(0))
        top5.update(prec5.item(), inputs.size(0))

    if args.dataset == 'mnist':
        if isTrain:
            for i in range(len(model.fc)):
                mu_G_list[i] /= sum(MNIST_TRAIN_SAMPLES)
                for j in range(len(MNIST_TRAIN_SAMPLES)):
                    mu_c_dict_list[i][j] /= MNIST_TRAIN_SAMPLES[j]
        else:
            for i in range(len(model.fc)):
                mu_G_list[i] /= sum(MNIST_TEST_SAMPLES)
                for j in range(len(MNIST_TEST_SAMPLES)):
                    mu_c_dict_list[i][j] /= MNIST_TEST_SAMPLES[j]
    elif args.dataset == 'cifar10' or args.dataset == 'cifar10_random':
        if isTrain:
            
            for i in range(len(model.fc)):
                mu_G_list[i] /= sum(CIFAR10_TRAIN_SAMPLES)
                for j in range(len(CIFAR10_TRAIN_SAMPLES)):
                    mu_c_dict_list[i][j] /= CIFAR10_TRAIN_SAMPLES[j]
        else:
            for i in range(len(model.fc)):
                mu_G_list[i] /= sum(CIFAR10_TEST_SAMPLES)
                for j in range(len(CIFAR10_TEST_SAMPLES)):
                    mu_c_dict_list[i][j] /= CIFAR10_TEST_SAMPLES[j]

    return mu_G_list, mu_c_dict_list, top1.avg, top5.avg


def compute_Sigma_W(args, model, fc_features, mu_c_dict, dataloader, isTrain=True):

    Sigma_W = 0
    for batch_idx, (inputs, targets) in enumerate(dataloader):

        inputs, targets = inputs.to(args.device), targets.to(args.device)

        with torch.no_grad():
            outputs = model(inputs)

        features = fc_features.outputs[0][0]
        fc_features.clear()

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


def compute_NC2_ETF_W(W):
    K = W.shape[0]
    WWT = torch.mm(W, W.T)
    WWT /= torch.norm(WWT, p='fro')

    sub = (torch.eye(K) - 1 / K * torch.ones((K, K))).cuda() / pow(K - 1, 0.5)
    ETF_metric = torch.norm(WWT - sub, p='fro')
    return ETF_metric.detach().cpu().numpy().item()


def compute_NC2_Identity_W(W):
    K = W.shape[0]
    WWT = torch.mm(W, W.T)
    WWT /= torch.norm(WWT, p='fro')
    sub = 1/pow(K, 0.5) * torch.eye(K).cuda()
    ETF_metric = torch.norm(WWT - sub, p='fro')
    return ETF_metric.detach().cpu().numpy().item()


def compute_NC2_ETF_H(mu_c_dict, mu_G):
    K = len(mu_c_dict)
    H = torch.empty(mu_c_dict[0].shape[0], K, device = mu_c_dict[0][0].device)
    for i in range(K):
        H[:, i] = mu_c_dict[i] - mu_G

    return compute_NC2_ETF_W(H.transpose(0,1))

def compute_NC2_Identity_H(mu_c_dict):
    K = len(mu_c_dict)
    H = torch.empty(mu_c_dict[0].shape[0], K, device = mu_c_dict[0][0].device)
    for i in range(K):
        H[:, i] = mu_c_dict[i]

    return compute_NC2_Identity_W(H.transpose(0,1))

def compute_NC3_ETF_WH(W, mu_c_dict, mu_G):
    K = len(mu_c_dict)
    H = torch.empty(mu_c_dict[0].shape[0], K, device = W.device)
    for i in range(K):
        H[:, i] = mu_c_dict[i] - mu_G

    WH = torch.mm(W, H.cuda())
    WH /= torch.norm(WH, p='fro')
    sub = 1 / pow(K - 1, 0.5) * (torch.eye(K) - 1 / K * torch.ones((K, K))).cuda()

    res = torch.norm(WH - sub, p='fro')
    return res.detach().cpu().numpy().item(), H

def compute_NC3_Identity_WH(W, mu_c_dict):
    K = len(mu_c_dict)
    H = torch.empty(mu_c_dict[0].shape[0], K, device = W.device)
    for i in range(K):
        H[:, i] = mu_c_dict[i]

    W_np = W.cpu().detach().numpy()
    H_np = H.cpu().detach().numpy()
    np.save("W_balance.npy", W_np)
    np.save("H_balance.npy", H_np)

    WH = torch.mm(W, H.cuda())
    WH /= torch.norm(WH, p='fro')
    sub = 1/pow(K, 0.5) * torch.eye(K).cuda()

    res = torch.norm(WH - sub, p='fro')
    return res.detach().cpu().numpy().item()

def compute_NC3_W_sub_H(W, mu_c_dict):
    K = len(mu_c_dict)
    H = torch.empty(mu_c_dict[0].shape[0], K, device = W.device)
    for i in range(K):
        H[:, i] = mu_c_dict[i]
    return torch.norm(W / torch.norm(W.cuda(), p='fro') - H.transpose(0,1) / torch.norm(H.cuda().transpose(0,1), p='fro'), p='fro')

def compute_NC3_W_sub_H_centered(W, mu_c_dict, mu_G):
    K = len(mu_c_dict)
    H = torch.empty(mu_c_dict[0].shape[0], K, device = W.device)
    for i in range(K):
        H[:, i] = mu_c_dict[i] - mu_G

    return torch.norm(W / torch.norm(W.cuda(), p='fro') - H.transpose(0,1) / torch.norm(H.cuda().transpose(0,1), p='fro'), p='fro')


def main():
    args = parse_eval_args()
    name = args.dataset + "-" + args.model \
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

    trainloader, testloader, num_classes = make_dataset(args.dataset, args.data_dir, args.batch_size, args.sample_size, SOTA=False)
    
    if args.model == "MLP":
        model = models.__dict__[args.model](hidden = args.width, depth_relu = args.depth_relu, depth_linear = args.depth_linear, fc_bias=args.bias, num_classes=num_classes).to(device)
    elif args.model.startswith("VGG"):
        model = models.__dict__[args.model](hidden = args.width, depth_linear = args.depth_linear, num_classes=num_classes, fc_bias=args.bias).to(device)
    elif args.model.startswith("ResNet"):
        model = models.__dict__[args.model](hidden = args.width, depth_linear = args.depth_linear, num_classes=num_classes, fc_bias=args.bias).to(device)
    elif args.model.startswith("ShuffleNetV2"):
        model = models.__dict__[args.model](hidden = args.width, depth_linear = args.depth_linear, num_classes=num_classes, fc_bias=args.bias).to(device)
    elif args.model.startswith("DenseNet"):
        model = models.__dict__[args.model](hidden = args.width, depth_linear = args.depth_linear, num_classes=num_classes, fc_bias=args.bias).to(device)

    fc_features = [FCFeatures() for i in range(len(model.fc))]
    for i in reversed(range(len(model.fc))):
        model.fc[i].register_forward_pre_hook(fc_features[len(model.fc) - 1 - i])

    info_dict = {
        'NC1_collapse_metric': [],
        'NC2_ETF_W': [],
        'NC2_Identity_W': [],
        'NC2_ETF_H': [],
        'NC3_ETF_WH': [],
        'NC3_W_sub_H': [],
        'W': [],
        'b': [],
        'H': [],
        'mu_G_train': [],
        'train_acc1': [],
        'train_acc5': [],
        'test_acc1': [],
        'test_acc5': []
    }

    for epoch in range(args.epochs):

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
                W_temp = W_temp @ model.fc[j].weight.clone()
                W_list.append(W_temp)
        if args.bias is True:
            b = model.fc[-1].bias
        mu_G_train_dict, mu_c_dict_train_dict, train_acc1, train_acc5 = compute_info(args, model, fc_features, trainloader, isTrain=True)
        mu_G_test_dict, mu_c_dict_test_dict, test_acc1, test_acc5 = compute_info(args, model, fc_features, testloader, isTrain=False)

        Sigma_W = compute_Sigma_W(args, model, fc_features[-1], mu_c_dict_train_dict[-1], trainloader, isTrain=True)
        Sigma_B = compute_Sigma_B(mu_c_dict_train_dict[-1], mu_G_train_dict[-1])

        NC1_collapse_metric = np.trace(Sigma_W @ scilin.pinv(Sigma_B)) / len(mu_c_dict_train_dict[-1])
        NC2_ETF_W = compute_NC2_ETF_W(W_list[-1])
        NC2_Identity_W = compute_NC2_Identity_W(W_list[-1])
        NC3_ETF_WH, H = compute_NC3_ETF_WH(W_list[-1], mu_c_dict_train_dict[-1], mu_G_train_dict[-1])
        info_dict['NC1_collapse_metric'].append(NC1_collapse_metric)
        if args.bias is True:
            info_dict['NC2_ETF_W'].append(NC2_ETF_W)
        else:
            info_dict['NC2_Identity_W'].append(NC2_Identity_W)
        info_dict['NC3_ETF_WH'].append(NC3_ETF_WH)

        info_dict['W'].append((W_list[0].detach().cpu().numpy()))
        if args.bias:
            info_dict['b'].append(b.detach().cpu().numpy())
        info_dict['H'].append(H.detach().cpu().numpy())

        info_dict['mu_G_train'].append(mu_G_train_dict[i].detach().cpu().numpy())

        info_dict['train_acc1'].append(train_acc1)
        info_dict['train_acc5'].append(train_acc5)
        info_dict['test_acc1'].append(test_acc1)
        info_dict['test_acc5'].append(test_acc5)
        
        for i in range(len(model.fc)):
            NC3_W_sub_H = compute_NC3_W_sub_H(W_list[i], mu_c_dict_train_dict[i])
            NC3_ETF_WH, _ = compute_NC3_ETF_WH(W_list[i], mu_c_dict_train_dict[i], mu_G_train_dict[i])
            NC2_Identity_W = compute_NC2_Identity_W(W_list[i])
            NC2_ETF_W = compute_NC2_ETF_W(W_list[i])

            NC3_Identity_WH = compute_NC3_Identity_WH(
                W_list[i], mu_c_dict=mu_c_dict_train_dict[i])
            NC3_W_sub_H_centered = compute_NC3_W_sub_H_centered(W_list[i], mu_c_dict_train_dict[i], mu_G_train_dict[i])
            NC2_ETF_H = compute_NC2_ETF_H(mu_c_dict_train_dict[i], mu_G_train_dict[i])
            NC2_Identity_H = compute_NC2_Identity_H(mu_c_dict_train_dict[i])

        print_and_save('[epoch: %d] | train top1: %.4f | train top5: %.4f | test top1: %.4f | test top5: %.4f ' %
                    (epoch + 1, train_acc1, train_acc5, test_acc1, test_acc5), None)



if __name__ == "__main__":
    main()