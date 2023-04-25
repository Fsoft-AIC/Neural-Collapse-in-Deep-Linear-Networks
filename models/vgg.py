'''VGG11/13/16/19 in Pytorch.'''
import torch
import torch.nn as nn


cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


class VGG(nn.Module):
    def __init__(self, vgg_name, hidden, depth_linear, fc_bias, num_classes):
        super(VGG, self).__init__()
        self.features = self._make_layers(cfg[vgg_name])
        self.batch_norm = nn.BatchNorm1d(num_features=512)
        if depth_linear > 1:
            fc = [nn.Linear(512, hidden, bias = False)]
            for i in range(depth_linear-2):
                fc += [nn.Linear(hidden, hidden, bias = False)]
            fc += [nn.Linear(hidden, num_classes, bias = fc_bias)]
        else:
            fc = [nn.Linear(512, num_classes, bias = fc_bias)]
        self.fc = nn.Sequential(*fc)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        features = self.batch_norm(out)
        out = self.fc(features)
        return out, features

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)


def test():
    net = VGG('VGG11')
    x = torch.randn(2,3,32,32)
    y = net(x)
    print(y.size())

def VGG11(hidden, depth_linear, fc_bias, num_classes=10):
    return VGG('VGG11', hidden, depth_linear, fc_bias, num_classes)

def VGG13(hidden, depth_linear, fc_bias, num_classes=10):
    return VGG('VGG13', hidden, depth_linear, fc_bias, num_classes)

def VGG16(hidden, depth_linear, fc_bias, num_classes=10):
    return VGG('VGG16', hidden, depth_linear, fc_bias, num_classes)

def VGG19(hidden, depth_linear, fc_bias, num_classes=10):
    return VGG('VGG19', hidden, depth_linear, fc_bias, num_classes)