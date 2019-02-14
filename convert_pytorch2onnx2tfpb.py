#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 22 15:18:10 2018

@author: pilgrim.bin@gmail.com
"""

import os
import random
import shutil

import numpy as np
from PIL import Image

# model
from diymodel import DIY_Model

# onnx - step 1
from torch.autograd import Variable
import torch.onnx

# onnx - step 2
import onnx
from onnx_tf.backend import prepare

# 
import tensorflow as tf

mlmc_tree = {
        'length': {'c5_changku': 4, 'c2_5fenku': 1, 'c1_duanku': 0, 'c3_7fenku': 2, 'c4_9fenku': 3}, 
        'style': {'F5_Denglong': 4, 'F7_Kuotui': 6, 'LT_Lianti': 9, 'F3_Zhitong': 2, 'LT_Beidai': 8, 'F4_Kuansong': 3, 'F2_Xiaojiao': 1, 'F8_Laba': 7, 'F6_Halun': 5, 'F1_JinshenQianbi': 0}}
#INFO: = mlmcdataloader.label_to_idx = {'length': 0, 'style': 1}

class_numbers = []
for key in sorted(mlmc_tree.keys()):
    class_numbers.append(len(mlmc_tree[key]))
    
print('------- = {}'.format(class_numbers))
    
def get_label_idx(label):
    idx = 0
    for key in mlmc_tree.keys():
        if label in mlmc_tree[key].keys():
            return idx
        idx += 1
    return None
    

# usage: is_allowed_extension(filename, IMG_EXTENSIONS)
IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif']
extensions = IMG_EXTENSIONS
def is_allowed_extension(filename, extensions):
    filename_lower = filename.lower()
    return any([filename_lower.endswith(ext) for ext in extensions])
    


def get_filelist(path):
    filelist = []
    for root,dirs,filenames in os.walk(path):
        for fn in filenames:
            this_path = os.path.join(root,fn)
            filelist.append(this_path)
    return filelist
   
# usage: mkdir_if_not_exist([root, dir])
def mkdir_if_not_exist(path):
    if not os.path.exists(os.path.join(*path)):
        os.makedirs(os.path.join(*path))
        

def get_dict_key(dict, value):
    for k in dict.keys():
        if dict[k] == value:
            return k
    return None

def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape((im_height, im_width, 3)).astype(np.uint8)


if __name__ == '__main__':
    
    # pipeline: pytorch model --> onnx modle --> tensorflow graph pb.
    
    # step 1, load pytorch model and export onnx during running.
    modelname = 'resnet18'
    weightfile = 'models/model_best_checkpoint_resnet18.pth.tar'
    modelhandle = DIY_Model(modelname, weightfile, class_numbers)
    model = modelhandle.model
    #model.eval() # useless
    dummy_input = Variable(torch.randn(1, 3, 224, 224)) # nchw
    onnx_filename = os.path.split(weightfile)[-1] + ".onnx"
    torch.onnx.export(model, dummy_input,
                      onnx_filename,
                      verbose=True)
    
    # step 2, create onnx_model using tensorflow as backend. check if right and export graph.
    onnx_model = onnx.load(onnx_filename)
    tf_rep = prepare(onnx_model, strict=False)
    # install onnx-tensorflow from github，and tf_rep = prepare(onnx_model, strict=False)
    # Reference https://github.com/onnx/onnx-tensorflow/issues/167
    #tf_rep = prepare(onnx_model) # whthout strict=False leads to KeyError: 'pyfunc_0'
    image = Image.open('pants.jpg')
    # debug, here using the same input to check onnx and tf.
    output_pytorch, img_np = modelhandle.process(image)
    print('output_pytorch = {}'.format(output_pytorch))
    output_onnx_tf = tf_rep.run(img_np)
    print('output_onnx_tf = {}'.format(output_onnx_tf))
    # onnx --> tf.graph.pb
    tf_pb_path = onnx_filename + '_graph.pb'
    tf_rep.export_graph(tf_pb_path)
    
    # step 3, check if tf.pb is right.
    with tf.Graph().as_default():
        graph_def = tf.GraphDef()
        with open(tf_pb_path, "rb") as f:
            graph_def.ParseFromString(f.read())
            tf.import_graph_def(graph_def, name="")
        with tf.Session() as sess:
            #init = tf.initialize_all_variables()
            init = tf.global_variables_initializer()
            #sess.run(init)
            
            # print all ops, check input/output tensor name.
            # uncomment it if you donnot know io tensor names.
            '''
            print('-------------ops---------------------')
            op = sess.graph.get_operations()
            for m in op:
                print(m.values())
            print('-------------ops done.---------------------')
            '''

            input_x = sess.graph.get_tensor_by_name("0:0") # input
            outputs1 = sess.graph.get_tensor_by_name('add_1:0') # 5
            outputs2 = sess.graph.get_tensor_by_name('add_3:0') # 10
            output_tf_pb = sess.run([outputs1, outputs2], feed_dict={input_x:img_np})
            #output_tf_pb = sess.run([outputs1, outputs2], feed_dict={input_x:np.random.randn(1, 3, 224, 224)})
            print('output_tf_pb = {}'.format(output_tf_pb))
   