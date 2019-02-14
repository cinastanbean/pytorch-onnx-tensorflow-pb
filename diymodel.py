#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 13 12:23:29 2018

@author: pilgrim.bin@gmail.com
"""
import os
import sys

import torch
import torchvision.models as models
import torchvision.transforms as transforms

import mlmcmodel

modelnames = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))


def base_model(modelname):
    for mn in modelnames:
        if mn in modelname:
            return mn
    return None


'''
    if saving the model using nn.DataParallel, which stores the model in module,
    we should convert the keys "module.***" -> "***" when trying to
    load it without DataParallel
'''
from collections import OrderedDict
def cvt_state_dict(state_dict):
    if not state_dict.keys()[0].startswith('module.'):
        return state_dict
    # create new OrderedDict that does not contain 'module'.
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] # remove `module.`
        new_state_dict[name] = v
    return new_state_dict



class DIY_Model():
    def __init__(self, modelname, weightfile, class_numbers, gpus=None):    
        # input check
        bm = base_model(modelname)
        if bm is None:
            raise(RuntimeError("Error: invalid modelname = {}".format(modelname)))
        if not os.path.exists(weightfile):
            raise(RuntimeError("Error: weightfile is not existed = {}".format(weightfile)))
        
        # create model @ both using in inception/renet18
        self.bm = bm
        self.model = models.__dict__[bm]()
        fc_features = self.model.fc.in_features
        self.model.fc = mlmcmodel.BuildMultiLabelModel(fc_features, class_numbers)
        
        '''
        if 'inception' in bm:
            # auxiliary fc
            aux_logits_fc_features = self.model.AuxLogits.fc.in_features
            self.model.AuxLogits.fc = nn.Linear(\
                aux_logits_fc_features, out_features=class_number, bias=True)
            '''
        if torch.cuda.is_available():
            self.model.cuda()
        else:
            self.model.cpu()
        
        # load model weight
        if torch.cuda.is_available():
            checkpoint = torch.load(weightfile)
        else:
            checkpoint = torch.load(weightfile, map_location='cpu')
        
        self.model.load_state_dict(
                cvt_state_dict(checkpoint['state_dict']))
        print("=> loaded checkpoint '{}'.".format(weightfile))
 
        # switch to evaluate mode
        self.model.eval()
        
        # preprocess transform
        
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        #normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.22, 0.22, 0.22])
        if 'inception' in bm:
            self.transform = transforms.Compose([
                transforms.Resize(320),
                transforms.CenterCrop(299),
                transforms.ToTensor(),
                normalize,
            ])
        else: # resnet18
            self.transform = transforms.Compose([
                transforms.Resize(224), # raw = 256 
                transforms.CenterCrop(224),
                #transforms.Resize(224),
                transforms.ToTensor(),
                normalize,
            ])
        

    def process(self, img):
        input = self.transform(img)
        
        if 'inception' in self.bm:
            input = input.reshape([1,3,299,299])
        else:
            input = input.reshape([1,3,224,224])
        
        if torch.cuda.is_available():
            output = self.model(input.cuda())
        else:
            output = self.model(input)
            
        result_list = []
        for i in range(len(output)):
            result_list.append(output[i][0].cpu().detach().numpy())
        
        # using for deployment
        #return result_list
    
        # using for onnx
        return (result_list, input.cpu().detach().numpy())
    
    
    
    
    