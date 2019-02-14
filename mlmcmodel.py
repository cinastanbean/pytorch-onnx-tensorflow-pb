#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 30 20:15:49 2018

@author: pilgrim.bin@gmail.com
"""
import torch.nn as nn
from torch.nn import init
from torch.nn import functional as F

class MultiLabelModel(nn.Module):
    def __init__(self, basemodel_output, num_classes, basemodel=None):
        super(MultiLabelModel, self).__init__()
        self.basemodel = basemodel
        self.num_classes = num_classes

        # config
        self.cfg_normalize = False # unchecked other method, diff with embedding.
        self.cfg_has_embedding = True
        self.cfg_num_features = basemodel_output # is there a better number?
        self.cfg_dropout_ratio = 0. #  0. is better than 0.8 at attributes:pants problem

        # diy head
        for index, num_class in enumerate(num_classes):
            if self.cfg_has_embedding:
                setattr(self, "EmbeddingFeature_FCLayer_" + str(index), nn.Linear(basemodel_output, self.cfg_num_features))
                setattr(self, "EmbeddingFeature_FCLayer_BN_" + str(index), nn.BatchNorm1d(self.cfg_num_features))
                feat = getattr(self, "EmbeddingFeature_FCLayer_" + str(index))
                feat_bn = getattr(self, "EmbeddingFeature_FCLayer_BN_" + str(index))
                init.kaiming_normal(feat.weight, mode='fan_out')
                init.constant(feat.bias, 0)
                init.constant(feat_bn.weight, 1)
                init.constant(feat_bn.bias, 0)
            if self.cfg_dropout_ratio > 0:
                setattr(self, "Dropout_" + str(index), nn.Dropout(self.cfg_dropout_ratio))
            setattr(self, "FullyConnectedLayer_" + str(index), nn.Linear(self.cfg_num_features, num_class))
            classifier = getattr(self, "FullyConnectedLayer_" + str(index))
            init.normal(classifier.weight, std=0.001)
            init.constant(classifier.bias, 0)
    
    def forward(self, x):
        if self.basemodel is not None:
            x = self.basemodel.forward(x)
        outs = list()
        for index, num_class in enumerate(self.num_classes):
            if self.cfg_has_embedding:
                feat = getattr(self, "EmbeddingFeature_FCLayer_" + str(index))
                feat_bn = getattr(self, "EmbeddingFeature_FCLayer_BN_" + str(index))
                x = feat(x)
                x = feat_bn(x)
            if self.cfg_normalize:
                x = F.normalize(x) # getattr bug
            elif self.cfg_has_embedding:
                x = F.relu(x)
            if self.cfg_dropout_ratio > 0:
                dropout = getattr(self, "Dropout_" + str(index))
                x = dropout(x)
            classifier = getattr(self, "FullyConnectedLayer_" + str(index))
            out = classifier(x)
            outs.append(out)
        return outs


def LoadPretrainedModel(model, pretrained_state_dict):
    model_dict = model.state_dict()
    union_dict = {k : v for k,v in pretrained_state_dict.iteritems() if k in model_dict}
    model_dict.update(union_dict)
    return model_dict

def BuildMultiLabelModel(basemodel_output, num_classes, basemodel=None):
    return MultiLabelModel(basemodel_output, num_classes, basemodel=basemodel)

'''----------------------------------------------------------------------------------------------------'''

# original version of https://github.com/pangwong/pytorch-multi-label-classifier.git
'''
import torch.nn as nn

class MultiLabelModel(nn.Module):
    def __init__(self, basemodel, basemodel_output, num_classes):
        super(MultiLabelModel, self).__init__()
        self.basemodel = basemodel
        self.num_classes = num_classes
        for index, num_class in enumerate(num_classes):
            setattr(self, "FullyConnectedLayer_" + str(index), nn.Linear(basemodel_output, num_class))
    
    def forward(self, x):
        x = self.basemodel.forward(x)
        outs = list()
        dir(self)
        for index, num_class in enumerate(self.num_classes):
            fun = eval("self.FullyConnectedLayer_" + str(index))
            out = fun(x)
            outs.append(out)
        return outs

def LoadPretrainedModel(model, pretrained_state_dict):
    model_dict = model.state_dict()
    union_dict = {k : v for k,v in pretrained_state_dict.iteritems() if k in model_dict}
    model_dict.update(union_dict)
    return model_dict

def BuildMultiLabelModel(basemodel, basemodel_output, num_classes):
    return MultiLabelModel(basemodel, basemodel_output, num_classes)

'''



