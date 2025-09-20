# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from calendar import c
from cmath import log
from itertools import count
from turtle import update
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
from torch.autograd import Variable
import torchvision.models as models

import random
import copy
import numpy as np
from collections import defaultdict, OrderedDict


from domainbed import networks
from domainbed.lib.misc import (
    random_pairs_of_minibatches, ParamDict, MovingAverage, l2_between_dicts
)
from domainbed.sam import SAM


ALGORITHMS = [
    'CDmix',
    'CDmix_sam',
    'ERM',
]

def get_algorithm_class(algorithm_name):
    if algorithm_name not in globals(): # globals store all the defined variables so far
        raise NotImplementedError("Algorithm not found: {}".format(algorithm_name))
    return globals()[algorithm_name]

class Algorithm(torch.nn.Module):
    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(Algorithm, self).__init__()
        self.hparams = hparams 

    def update(self, minibatches, unlabeled=None):
        raise NotImplementedError

    def predict(self, x):
        raise NotImplementedError

class ERM(Algorithm):
    """
    Empirical Risk Minimization (ERM)
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(ERM, self).__init__(input_shape, num_classes, num_domains,
                                  hparams)
        self.hparams = hparams
        self.input_shape = input_shape 
        self.num_classes = num_classes
        self.num_domains = num_domains
        self.batch_size = hparams['batch_size']
        self.network = networks.ResNet_DomainClassMixUp(input_shape, num_classes, num_domains, hparams)

        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=0.001,
            weight_decay=0.0001
        )
        self.domain_optimizer = torch.optim.Adam(
                self.network.domain_classifier.parameters(),
                lr=0.001,
                weight_decay=0.0001
            )

    def update(self, minibatches, unlabeled=None):
        all_x = torch.cat([x for x,y in minibatches])
        all_y = torch.cat([y for x,y in minibatches])
        all_domain = torch.LongTensor(self.batch_size*3).to(all_x.device)
        for i in range(self.num_domains):
            all_domain[i*self.batch_size:(i+1)*self.batch_size] = i
        loss = F.cross_entropy(self.predict(all_x), all_y)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        features = self.network.get_feature(all_x).detach()
        result = self.network.domain_classifier(features)
        loss_domain = F.cross_entropy(result, all_domain)
        self.domain_optimizer.zero_grad()
        loss_domain.backward()
        self.domain_optimizer.step()
        return {'loss': loss.item()}

    def predict(self, x):
        return self.network(x)
    
    def get_feature(self, x):
        return self.network.get_feature(x)

class CDmix(nn.Module):
    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(CDmix, self).__init__()
        self.hparams = hparams
        self.input_shape = input_shape 
        self.num_classes = num_classes
        self.num_domains = num_domains
        self.batch_size = 32

        self.network = networks.ResNet_DomainClassMixUp(input_shape, num_classes, num_domains, hparams)
        self.optimizer = torch.optim.Adam(
                self.network.parameters(),
                lr=0.001,
                weight_decay=0.0001
            ) # lr=self.hparams["lr"],
        self.domain_optimizer = torch.optim.Adam(
                self.network.domain_classifier.parameters(),
                lr=0.001,
                weight_decay=0.0001
            )

        self.domain_gradient = None
        self.class_gradient = None 
        
        self.class_target_layer = self.network.maxpool2
        self.register_gradient()

        self.call_num = 0
    
    def register_gradient(self):
        def save_gradient(module, input, output):
            if not hasattr(output, "requires_grad") or not output.requires_grad:
                return
            def _store_grad(grad):
                self.class_gradient = grad.cpu().detach()
            output.register_hook(_store_grad)
        self.class_target_layer.register_forward_hook(save_gradient)

    def update(self, minibatches, unlabeled=None):
        all_x = torch.cat([x for x,y in minibatches])
        all_y = torch.cat([y for x,y in minibatches])
        all_domain = torch.LongTensor(self.batch_size*self.num_domains).to(all_x.device) 
        for i in range(self.num_domains):  
            all_domain[i*self.batch_size:(i+1)*self.batch_size] = i
       
        if self.call_num < self.hparams['warmup_step']: 
            loss_class = F.cross_entropy(self.predict(all_x), all_y)
            self.optimizer.zero_grad()
            loss_class.backward()
            self.optimizer.step()
        else:
            if self.call_num == self.hparams['warmup_step']:
                print('start mixup')
            self.class_gradient = None 
            prediction_output = self.predict(all_x) # 96, 19 
            loss = sum([prediction_output[i][all_y[i].item()] for i in range(all_y.size(0))])
            self.optimizer.zero_grad()
            loss.backward(retain_graph=True)
            # 计算域分类梯度
            self.domain_gradient = None    
            features = self.network.get_feature(all_x).detach()  # 96,64,27,12
            features.requires_grad_(True)
            result = self.network.domain_classifier(features) # 96,3
            loss = sum([result[i][all_domain[i].item()] for i in range(all_domain.size(0))])
            self.domain_optimizer.zero_grad()
            loss.backward(retain_graph=True)
            self.domain_gradient = features.grad
            # 混合更新
            loss1 = F.cross_entropy(prediction_output, all_y)  # 原始损失
            loss2 = F.cross_entropy(self.network.train_f(all_x, all_y, all_domain, self.class_gradient, self.domain_gradient), all_y)
            # 基于梯度的增强分类损失
            loss_class = 0.5*loss1+0.5*loss2
            self.optimizer.zero_grad()
            loss_class.backward()
            self.optimizer.step()
        # 提取特征并计算域分类结果，计算域分类损失并进行优化步骤。
        features = self.network.get_feature(all_x).detach()
        result = self.network.domain_classifier(features)
        loss_domain = F.cross_entropy(result, all_domain)
        self.domain_optimizer.zero_grad()
        loss_domain.backward()
        self.domain_optimizer.step()
        # 更新调用次数，返回域分类损失和任务分类损失。
        self.call_num += 1
        return {'loss_domain': loss_domain.item(), 'loss_task': loss_class.item()}

        
    def predict(self, x):
        return self.network(x)
    
    def get_feature(self, x):
        return self.network.get_feature(x)


class CDmix_sam(nn.Module):
    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(CDmix_sam, self).__init__(input_shape, num_classes, num_domains, hparams)
        self.hparams = hparams
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.num_domains = num_domains
        self.batch_size = hparams['batch_size']

        self.network = networks.ResNet_DomainClassMixUp(input_shape, num_classes, num_domains, hparams)
        base_optimizer = torch.optim.Adam
        self.optimizer = SAM(
            self.network.network.parameters(),
            base_optimizer, 
            lr=self.hparams["lr"],
            weight_decay=self.hparams['weight_decay']
        )
        self.domain_optimizer = torch.optim.Adam(
                self.network.domain_classifier.parameters(),
                lr=self.hparams["lr"],
                weight_decay=self.hparams['weight_decay']
            )

        self.domain_gradient = None
        self.class_gradient = None 
        
        self.class_target_layer = self.network.network.conv2
        self.register_gradient()

        self.call_num = 0

    def register_gradient(self):
        def save_gradient(module, input, output):
            if not hasattr(output, "requires_grad") or not output.requires_grad:
                return
            def _store_grad(grad):
                self.class_gradient = grad.cpu().detach()
            output.register_hook(_store_grad)
        self.class_target_layer.register_forward_hook(save_gradient)

    def update(self, minibatches, unlabeled=None):
        all_x = torch.cat([x for x,y in minibatches])
        all_y = torch.cat([y for x,y in minibatches])
        all_domain = torch.LongTensor(self.batch_size*self.num_domains).to(all_x.device)
        for i in range(self.num_domains):
            all_domain[i*self.batch_size:(i+1)*self.batch_size] = i
        
        if self.call_num < self.hparams['warmup_step']:
            loss_class = F.cross_entropy(self.predict(all_x), all_y)
            loss_class.backward()
            self.optimizer.first_step(zero_grad=True)
            
            loss_class2 = F.cross_entropy(self.predict(all_x), all_y)
            loss_class2.backward()
            self.optimizer.second_step(zero_grad=True)
        else:
            if self.call_num == self.hparams['warmup_step']:
                print('start mixup')
            # get class gradients
            self.class_gradient = None 
            prediction_output = self.predict(all_x)
            loss = sum([prediction_output[i][all_y[i].item()] for i in range(all_y.size(0))])
            self.optimizer.zero_grad()
            loss.backward(retain_graph=True)
            # get domain gradients
            self.domain_gradient = None    
            features = self.network.get_feature(all_x).detach()
            features.requires_grad_(True)
            result = self.network.domain_classifier(features)
            loss = sum([result[i][all_domain[i].item()] for i in range(all_domain.size(0))])
            self.domain_optimizer.zero_grad()
            loss.backward(retain_graph=True)
            self.domain_gradient = features.grad
            # augmentation and update 
            # 特征增强后计算的损失，反映了增强后的特征对分类任务的贡献。通过这种方式，可以同时优化模型在原始特征和增强特征上的性能，提高模型的泛化能力
            loss1 = F.cross_entropy(prediction_output, all_y)
            loss2 = F.cross_entropy(self.network.train_f(all_x, all_y, all_domain, self.class_gradient, self.domain_gradient), all_y)
            loss_class = 0.5*loss1+0.5*loss2
            loss_class.backward()
            self.optimizer.first_step(zero_grad=True)

            loss11 = F.cross_entropy(self.predict(all_x), all_y)
            loss22 = F.cross_entropy(self.network.train_f(all_x, all_y, all_domain, self.class_gradient, self.domain_gradient), all_y)
            loss_class2 = 0.5*loss11+0.5*loss22
            loss_class2.backward()
            self.optimizer.second_step(zero_grad=True)



        features = self.network.get_feature(all_x).detach()
        result = self.network.domain_classifier(features)
        loss_domain = F.cross_entropy(result, all_domain)
        self.domain_optimizer.zero_grad()
        loss_domain.backward()
        self.domain_optimizer.step()
        
        self.call_num += 1
        return {'loss_domain': loss_domain.item(), 'loss_task': loss_class.item()}

        
    def predict(self, x):
        return self.network(x)
