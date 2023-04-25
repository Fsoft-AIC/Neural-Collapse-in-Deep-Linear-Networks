import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, hidden, depth_relu = 6, depth_linear=1, fc_bias = True, num_classes = 10, batchnorm=True):
        
        super(MLP, self).__init__()
        self.batch_norm_flag = batchnorm
        if batchnorm:
            layers = [nn.Linear(3072, hidden), nn.ReLU(), nn.BatchNorm1d(num_features=hidden)]
        else:
            layers = [nn.Linear(3072, hidden), nn.ReLU()]
        
        for i in range(depth_relu - 1):
            if batchnorm:
                layers += [nn.Linear(hidden, hidden), nn.ReLU(), nn.BatchNorm1d(num_features=hidden)]
            else:
                layers += [nn.Linear(hidden, hidden), nn.ReLU()]

        self.layers = nn.Sequential(*layers)
        for i in range(depth_relu):
            if isinstance(self.layers[i], nn.Linear):
                nn.init.orthogonal_(self.layers[i].weight)
        if batchnorm is True:
            self.batch_norm = nn.BatchNorm1d(num_features=hidden)
        else:
            self.intermediate_layer = nn.Linear(hidden, hidden, bias = True)
        fc = []
        for i in range(depth_linear-1):
            fc += [nn.Linear(hidden, hidden, bias = False)]
        fc += [nn.Linear(hidden, num_classes, bias = fc_bias)]
        self.fc = nn.Sequential(*fc)
        for i in range(depth_linear):
            if isinstance(self.fc[i], nn.Linear):
                nn.init.orthogonal_(self.fc[i].weight)

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x = self.layers(x)
        if self.batch_norm_flag:
            x = self.batch_norm(x)
        else:
            x = self.intermediate_layer(x)
        features = x
        x = self.fc(features)
        return x, features
