# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import numpy as np 
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models

from collections import OrderedDict

from domainbed.lib import wide_resnet
from domainbed import mixup_module
import copy

def remove_batch_norm_from_resnet(model):
    fuse = torch.nn.utils.fusion.fuse_conv_bn_eval
    model.eval()

    model.conv1 = fuse(model.conv1, model.bn1)
    model.bn1 = Identity()

    for name, module in model.named_modules():
        if name.startswith("layer") and len(name) == 6:
            for b, bottleneck in enumerate(module):
                for name2, module2 in bottleneck.named_modules():
                    if name2.startswith("conv"):
                        bn_name = "bn" + name2[-1]
                        setattr(bottleneck, name2,
                                fuse(module2, getattr(bottleneck, bn_name)))
                        setattr(bottleneck, bn_name, Identity())
                if isinstance(bottleneck.downsample, torch.nn.Sequential):
                    bottleneck.downsample[0] = fuse(bottleneck.downsample[0],
                                                    bottleneck.downsample[1])
                    bottleneck.downsample[1] = Identity()
    model.train()
    return model


class Identity(nn.Module):
    """An identity layer"""
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class MLP(nn.Module):
    """Just an MLP"""
    def __init__(self, n_inputs, n_outputs, hparams):
        super(MLP, self).__init__()
        self.input = nn.Linear(n_inputs, hparams['mlp_width'])
        self.dropout = nn.Dropout(hparams['mlp_dropout'])
        self.hiddens = nn.ModuleList([
            nn.Linear(hparams['mlp_width'], hparams['mlp_width'])
            for _ in range(hparams['mlp_depth']-2)])
        self.output = nn.Linear(hparams['mlp_width'], n_outputs)
        self.n_outputs = n_outputs

    def forward(self, x):
        x = self.input(x)
        x = self.dropout(x)
        x = F.relu(x)
        for hidden in self.hiddens:
            x = hidden(x)
            x = self.dropout(x)
            x = F.relu(x)
        x = self.output(x)
        return x

# 在训练过程中引入分布不确定性，以增强模型的鲁棒性和泛化能力
class DistributionUncertainty(nn.Module):
    """
    Distribution Uncertainty Module
        Args:
        p   (float): probabilty of foward distribution uncertainty module, p in [0,1].

    """

    def __init__(self, p=0.5, eps=1e-6):
        super(DistributionUncertainty, self).__init__()
        self.eps = eps
        self.p = p
        self.factor = 1.0

    def _reparameterize(self, mu, std):
        epsilon = torch.randn_like(std) * self.factor
        return mu + epsilon * std

    def sqrtvar(self, x):
        t = (x.var(dim=0, keepdim=True) + self.eps).sqrt()
        t = t.repeat(x.shape[0], 1)
        return t

    def forward(self, x):
        if (not self.training) or (np.random.random()) > self.p:
            return x

        mean = x.mean(dim=[2, 3], keepdim=False)
        std = (x.var(dim=[2, 3], keepdim=False) + self.eps).sqrt()

        sqrtvar_mu = self.sqrtvar(mean)
        sqrtvar_std = self.sqrtvar(std)

        beta = self._reparameterize(mean, sqrtvar_mu)
        gamma = self._reparameterize(std, sqrtvar_std)

        x = (x - mean.reshape(x.shape[0], x.shape[1], 1, 1)) / std.reshape(x.shape[0], x.shape[1], 1, 1)
        x = x * gamma.reshape(x.shape[0], x.shape[1], 1, 1) + beta.reshape(x.shape[0], x.shape[1], 1, 1)

        return x

class DomainLearner(nn.Module):
    def __init__(self,  num_domain):
        super(DomainLearner, self).__init__()
        self.n_outputs = 64
        
        self.featurizer = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(self.n_outputs, num_domain)
        
        self.network = nn.Sequential(self.featurizer, self.classifier)
        # self.freeze_bn()

    def forward(self, x):
        x = self.featurizer(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x 
    
    def train(self, mode=True):
        """
        Override the default train() to freeze the BN parameters
        """
        super().train(mode)
        # self.freeze_bn()
        return self 


class ResNet_DomainClassMixUp(nn.Module):
    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(ResNet_DomainClassMixUp, self).__init__()
        self.num_classes = 19
        self.num_domains = 3
        self.conv1 = nn.Conv2d(1, 32, (9,1), 1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, (9,1), stride=1, padding=1)

        self.bn0 = nn.BatchNorm2d(32)
        self.bn1 = nn.BatchNorm2d(64)

        self.maxpool1 = nn.MaxPool2d(kernel_size=(1, 2), stride=2)
        self.maxpool2 = nn.MaxPool2d(kernel_size=(1, 2), stride=2)
        self.fc = nn.Linear(20736, self.num_classes)

        self.mixup_module = mixup_module.DomainClassMixAugmentation(hparams['batch_size'], num_classes, num_domains, hparams)
        self.domain_classifier = DomainLearner(self.num_domains)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn0(x)
        x = F.relu(x)
        x = self.maxpool1(x)
        x = self.conv2(x)       
        x = self.bn1(x)
        x = F.relu(x)
        x = self.maxpool2(x)
        x = x.view(x.shape[0],-1)
        x = self.fc(x)

        return x

    def train_f(self, x, y, domain, class_gradient, domain_gradient):
        x = self.conv1(x)
        x = self.bn0(x)
        x = F.relu(x)
        x = self.maxpool1(x)
        x = self.conv2(x)       
        x = self.bn1(x)
        x = F.relu(x)
        x = self.maxpool2(x)
        x = self.mixup_module.forward(x, y, domain, class_gradient, domain_gradient)
        x = x.view(x.shape[0],-1)
        x = self.fc(x)
        return x
    
    def get_feature(self, x):
        x = self.conv1(x)
        x = self.bn0(x)
        x = F.relu(x)
        x = self.maxpool1(x)
        x = self.conv2(x)       
        x = self.bn1(x)
        x = F.relu(x)
        x = self.maxpool2(x)
        return x 

    def get_whole_feature(self, x):
        x = self.get_feature(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return x 
    
    def train(self, mode=True):
        """
        Override the default train() to freeze the BN parameters
        """
        super().train(mode)
        return self 

