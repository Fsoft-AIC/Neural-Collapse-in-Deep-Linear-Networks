import torch
import torch.nn as nn
import argparse
import numpy as np
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lrs
from tqdm import tqdm
import scipy.linalg as scilin
import utils
import utils
import math
from sympy import Symbol, solve, S

NUM_CLASS_SAMPLES = [100, 100, 100, 100]
NUM_SAMPLES = sum(NUM_CLASS_SAMPLES)

class SyntheticModel(nn.Module):
    def __init__(self, hidden, num_classes, depth_linear, NUM_SAMPLES, fc_bias, weight_decay, feature_decay):
        super().__init__()
        self.weight_decay = weight_decay
        self.feature_decay = feature_decay
        self.depth_linear = depth_linear
        self.H = nn.Parameter(torch.normal(0, 0.1, size = (NUM_SAMPLES, hidden)), requires_grad=True)
        fc = []
        if depth_linear > 1:
            for i in range(depth_linear-1):
                fc += [nn.Linear(hidden, hidden, bias = False)]
            fc += [nn.Linear(hidden, num_classes, bias = fc_bias)]
        else:
            fc += [nn.Linear(hidden, num_classes, bias = fc_bias)]
        self.fc = nn.Sequential(*fc)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.1)
        
        
    def forward(self, Y):
        features_list = [self.H.detach().clone()]
        out = self.fc[0](self.H)
        if len(self.fc) > 1:
            features_list.append(out.detach().clone())
            for i in range(1, len(self.fc)-1):
                out = self.fc[i](out)
                features_list.append(out.detach().clone())
            out = self.fc[len(self.fc)-1](out)
        loss = torch.norm(out - Y, p='fro') ** 2 / (2*NUM_SAMPLES)
        loss += self.weight_decay / 2 * sum([torch.sum(self.fc[i].weight**2) for i in range(self.depth_linear)])
        loss += self.feature_decay / 2 * torch.sum(self.H ** 2)
        features_list.reverse()
        return out, loss, features_list

def compute_accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def compute_info(model, Y, targets):

    mu_G_list = [0] * (len(model.fc))
    pairs = [(i, 0) for i in range(len(NUM_CLASS_SAMPLES))]
    mu_c_dict_list = [dict(pairs) for i in range(len(model.fc))]
    outputs, _, features_list = model(Y)
    for i in range(len(model.fc)):
        for y in range(len(NUM_CLASS_SAMPLES)):
            indexes = (targets == y).nonzero(as_tuple=True)[0]
            if indexes.nelement()==0:
                mu_c_dict_list[y] = 0
            else:
                mu_c_dict_list[i][y] = features_list[i][indexes, :].sum(dim=0)
                mu_G_list[i] += mu_c_dict_list[i][y]
                mu_c_dict_list[i][y] = mu_c_dict_list[i][y] / NUM_CLASS_SAMPLES[y]
        mu_G_list[i] = mu_G_list[i] / sum(NUM_CLASS_SAMPLES)
        
    prec1 = compute_accuracy(outputs.data, targets.data, topk=(1,))
    return mu_G_list, mu_c_dict_list, prec1[0].cpu().numpy(), features_list

def compute_Sigma_W(H, mu_c_dict, targets):
    Sigma_W = 0
    for y in range(len(NUM_CLASS_SAMPLES)):
        indexes = (targets == y).nonzero(as_tuple=True)[0]
        if indexes.nelement()==0:
            Sigma_W += 0
        else:
            Sigma_W += ((H[indexes, :] - mu_c_dict[y]).unsqueeze(2) @ (H[indexes, :] - mu_c_dict[y]).unsqueeze(1)).sum(0)
    Sigma_W /= NUM_SAMPLES
    return Sigma_W.cpu().numpy()

def compute_Sigma_B(mu_c_dict, mu_G):
    Sigma_B = 0
    K = len(mu_c_dict)
    for i in range(K):
        Sigma_B += (mu_c_dict[i] - mu_G).unsqueeze(1) @ (mu_c_dict[i] - mu_G).unsqueeze(0)

    Sigma_B /= K

    return Sigma_B.cpu().numpy()

#Computing the c*s
def compute_s(weight_decay, feature_decay_rate, depth, device):
    c = 1
    s_array = torch.zeros(len(NUM_CLASS_SAMPLES), device=device)

    x = compute_x(NUM_CLASS_SAMPLES, depth, weight_decay, feature_decay_rate, device=device)
    s_array = pow(NUM_SAMPLES*feature_decay_rate *
                  pow(x, depth) / c, 1/(2*depth))

    return s_array


def compute_x(NUM_CLASS_SAMPLES: list, M: float, lambda_weight: float, lambda_feature: float, device):
    N = sum(NUM_CLASS_SAMPLES)
    b = M * N * math.pow(N * math.pow(lambda_weight, M) * lambda_feature, 1/M)
    x = Symbol('x', real=True)
    result = torch.empty(len(NUM_CLASS_SAMPLES), device=device)
    for i in range(len(NUM_CLASS_SAMPLES)):
        result[i] = torch.from_numpy(np.array(max(solve(b / NUM_CLASS_SAMPLES[i] - (M * x**(M-1))/((x ** M + 1)**2), x, rational=False)), dtype=np.float32)).to(device)
    return result


def compute_NC2_W(W, s_array, depth):
    K = W.shape[0]
    WWT = torch.mm(W, W.T)
    WWT /= torch.norm(WWT, p='fro')
    diag_array = torch.pow(s_array, 2* depth)
    sub = torch.diag_embed(diag_array)
    sub /= torch.norm(sub, p='fro')
    metric = torch.norm(WWT - sub, p='fro')
    return metric.detach().cpu().numpy().item()

def compute_NC2_H(mu_c_dict, s_array, feature_decay_rate, depth):
    K = len(mu_c_dict)
    H = torch.empty(mu_c_dict[0].shape[0], K, device = s_array.device)
    for i in range(K):
        H[:, i] = mu_c_dict[i]
    HTH = torch.mm(H.T, H)
    s_array_square_M = pow(s_array, 2*depth)
    diag_array = torch.div(s_array_square_M, 
                           pow((s_array_square_M + NUM_SAMPLES * feature_decay_rate), 2))
    sub = torch.diag_embed(diag_array)
    metric = torch.norm(HTH - sub, p='fro')
    return metric.detach().cpu().numpy().item()

def compute_NC3_WH(W, mu_c_dict, s_array, feature_decay_rate, depth):
    K = len(mu_c_dict)
    H = torch.empty(mu_c_dict[0].shape[0], K, device = W.device)
    for i in range(K):
        H[:, i] = mu_c_dict[i]
    s_array_square_M = pow(s_array, 2*depth)
    diag_array = torch.div(s_array_square_M, 
                           s_array_square_M + NUM_SAMPLES * feature_decay_rate)
    WH = torch.mm(W, H.cuda())
    WH /= torch.norm(WH, p='fro')
    sub = torch.diag_embed(diag_array)
    sub /= torch.norm(sub, p='fro')
    res = torch.norm(WH - sub, p='fro')
    return res.detach().cpu().numpy().item()

def compute_NC3_W_sub_H(W, mu_c_dict, s_array, feature_decay_rate, depth):
    K = len(mu_c_dict)
    H = torch.empty(mu_c_dict[0].shape[0], K, device = W.device)
    diag_array = torch.zeros(len(NUM_CLASS_SAMPLES), device = W.device)
    for i in range(K):
        H[:, i] = mu_c_dict[i]
        diag_array[i] = s_array[i]**(2*depth) + NUM_SAMPLES*feature_decay_rate
    
    sub = torch.diag_embed(diag_array) @ H.T
    res = torch.norm(W - sub, p='fro')
    return res.detach().cpu().numpy().item()

def compute_NC2_ETF_W(W):
    K = W.shape[0]
    WWT = torch.mm(W, W.T)
    WWT /= torch.norm(WWT, p='fro')

    sub = (torch.eye(K) - 1 / K * torch.ones((K, K))).cuda() / pow(K - 1, 0.5)
    ETF_metric = torch.norm(WWT - sub, p='fro')
    return ETF_metric.detach().cpu().numpy().item()

def compute_NC3_ETF_WH(W, mu_c_dict, mu_G):
    K = len(mu_c_dict)
    H = torch.empty(mu_c_dict[0].shape[0], K, device = W.device)
    for i in range(K):
        H[:, i] = mu_c_dict[i] - mu_G

    WH = torch.mm(W, H.cuda())
    WH /= torch.norm(WH, p='fro')
    sub = 1 / pow(K - 1, 0.5) * (torch.eye(K) - 1 / K * torch.ones((K, K))).cuda()

    res = torch.norm(WH - sub, p='fro')
    return res.detach().cpu().numpy().item()

def compute_NC2_Identity_W(W):
    K = W.shape[0]
    WWT = torch.mm(W, W.T)
    WWT /= torch.norm(WWT, p='fro')
    sub = 1/pow(K, 0.5) * torch.eye(K).cuda()
    ETF_metric = torch.norm(WWT - sub, p='fro')
    return ETF_metric.detach().cpu().numpy().item()


def compute_NC3_Identity_WH(W, mu_c_dict):
    K = len(mu_c_dict)
    H = torch.empty(mu_c_dict[0].shape[0], K, device = W.device)
    for i in range(K):
        H[:, i] = mu_c_dict[i]

    WH = torch.mm(W, H.cuda())
    WH /= torch.norm(WH, p='fro')
    sub = 1/pow(K, 0.5) * torch.eye(K).cuda()

    res = torch.norm(WH - sub, p='fro')
    return res.detach().cpu().numpy().item()

def main(args):
    utils.set_seed(args.seed)
    name = "balance-MSE" + "-bias_" + str(args.bias) + "-GOF_" + str(args.GOF) \
           + "-depth_linear_" + str(args.depth_linear) \
           + "-" + "lr_" + str(args.lr) \
           + "-" + "dim_" + str(args.hidden) \
           + "-" + "epochs_" + str(args.num_iteration) \
           + "-" + "seed_" + str(args.seed)

    S_ARRAY = compute_s(args.weight_decay, args.feature_decay, args.depth_linear, device="cuda")
    model = SyntheticModel(hidden=args.hidden, num_classes=args.num_classes, fc_bias=args.bias,
                           depth_linear=args.depth_linear, NUM_SAMPLES=NUM_SAMPLES,
                           weight_decay=args.weight_decay, feature_decay=args.feature_decay).to("cuda")
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)

    targets = []

    for idx in range(len(NUM_CLASS_SAMPLES)):
        targets = [*targets, *[idx]*NUM_CLASS_SAMPLES[idx]]
    targets = torch.tensor(targets).to("cuda")
    # Randomy shuffle
    targets = targets[torch.randperm(targets.shape[0])]
    Y = F.one_hot(targets).to("cuda").to(torch.float)
    pbar = tqdm(range(args.num_iteration))
    for iteration in pbar:
        model.train()
        optimizer.zero_grad()
        _, loss, features = model(Y)
        loss.backward()
        pbar.set_description("Loss is {:.6f}".format(loss))
        optimizer.step()
        with torch.no_grad():
            model.eval()
            W_list = []
            W_temp = model.fc[-1].weight.clone()
            W_list.append(W_temp)
            for j in list(reversed(range(0, len(model.fc)-1))):
                if isinstance(model.fc[j], nn.Linear):
                    W_temp = W_temp @ model.fc[j].weight
                    W_list.append(W_temp)
            mu_G_train_dict, mu_c_dict_train_dict, train_acc1, features_list = compute_info(model, Y, targets)
            Sigma_W = compute_Sigma_W(features_list[-1], mu_c_dict_train_dict[-1], targets)
            Sigma_B = compute_Sigma_B(mu_c_dict_train_dict[-1], mu_G_train_dict[-1])
            NC1_collapse_metric = np.trace(Sigma_W @ scilin.pinv(Sigma_B)) / len(mu_c_dict_train_dict[-1])
            if args.GOF is True:
                if args.bias is False:
                    NC2_W = compute_NC2_W(W_list[-1], S_ARRAY, args.depth_linear)
                    NC2_H = compute_NC2_H(mu_c_dict_train_dict[-1], S_ARRAY, args.feature_decay, args.depth_linear) #H only
                    NC3_WH = compute_NC3_WH(W_list[0], mu_c_dict_train_dict[0], S_ARRAY, args.feature_decay, args.depth_linear)
                    NC3_W_sub_H = compute_NC3_W_sub_H(W_list[-1], mu_c_dict_train_dict[-1], S_ARRAY, args.feature_decay, args.depth_linear)
            else:
                if args.bias is False:
                    for i in range(len(model.fc)):
                        NC2_ETF_W = compute_NC2_Identity_W(W_list[i])
                        NC3_ETF_WH = compute_NC3_Identity_WH(W_list[i], mu_c_dict_train_dict[i])
                else:
                    for i in range(len(model.fc)):
                        NC2_ETF_W = compute_NC2_ETF_W(W_list[i])
                        NC3_ETF_WH = compute_NC3_ETF_WH(W_list[i], mu_c_dict_train_dict[i], mu_G_train_dict[i])



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', type=int, default=4)
    parser.add_argument('--hidden', type=int, default=64)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--feature_decay', type=float, default=5e-4)
    parser.add_argument('--depth_linear', type=int, default=1)
    parser.add_argument('--num_iteration', type=int, default=30000)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--seed', type=str, default=2)
    parser.add_argument('--no-bias', dest='bias', action='store_false')
    parser.add_argument('--GOF', dest='GOF', action='store_true')
    args = parser.parse_args()
    main(args)