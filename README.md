# Converting A PyTorch Model to Tensorflow pb using ONNX

<p align="right">pilgrim.bin@gmail.com</p>

**有必要说在前面，避免后来者陷坑：** 

**ONNX本来是Facebook联合AWS对抗Tensorflow的，所以注定ONNX-TF这件事是奸情，这是ONNX和TF偷情的行为，两个平台都不会为他们背书；Pytorch和Tensorflow各自在独立演变，动态图和静态图优化两者不会停战。如果你在尝试转模型这件事情，觉得你有必要考虑：1.让服务部署平台支持Pytorch; 2.转训练平台到TF; 3.这件事是一锤子买卖，干完就不要再倒腾了。**； 

本Demo所使用模型来自：https://github.com/cinastanbean/Pytorch-Multi-Task-Multi-class-Classification


[TOC]

# 1. Pre-installation

**Version Info**

```
pytorch                   0.4.0           py27_cuda0.0_cudnn0.0_1    pytorch
torchvision               0.2.1                    py27_1    pytorch
tensorflow                1.8.0                     <pip>
onnx                      1.2.2                     <pip>
onnx-tf                   1.1.2                     <pip> 
```

注意：

1. ONNX1.1.2版本太低会引发BatchNormalization错误，当前pip已经支持1.3.0版本；也可以考虑源码安装 `pip install -U git+https://github.com/onnx/onnx.git@master`。
2. 本实验验证ONNX1.2.2版本可正常运行
3. onnx-tf采用源码安装；要求 Tensorflow>=1.5.0.；


# 2. 转换过程

## 2.1 Step 1.2.3.

**pipeline: pytorch model --> onnx modle --> tensorflow graph pb.**

```
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
```


## 2.2 Verification

**确保输出结果一致**

```
output_pytorch = [array([ 2.5359073 , -1.4261041 , -5.2394    , -0.62402934,  4.7426634 ], dtype=float32), array([ 7.6249304,  5.1203837,  1.8118637,  1.5143847, -4.9409146, 1.1695148, -6.2375665, -1.6033885, -1.4286405, -2.964429 ], dtype=float32)]
      
output_onnx_tf = Outputs(_0=array([[ 2.5359051, -1.4261056, -5.239397 , -0.6240269,  4.7426634]], dtype=float32), _1=array([[ 7.6249285,  5.12038  ,  1.811865 ,  1.5143874, -4.940915 , 1.1695154, -6.237564 , -1.6033876, -1.4286422, -2.964428 ]], dtype=float32))
      
output_tf_pb = [array([[ 2.5359051, -1.4261056, -5.239397 , -0.6240269,  4.7426634]], dtype=float32), array([[ 7.6249285,  5.12038  ,  1.811865 ,  1.5143874, -4.940915 , 1.1695154, -6.237564 , -1.6033876, -1.4286422, -2.964428 ]], dtype=float32)]
```

**独立TF验证程序**


```
def get_img_np_nchw(filename):
    try:
        image = Image.open(filename).convert('RGB').resize((224, 224))
        miu = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        #miu = np.array([0.5, 0.5, 0.5])
        #std = np.array([0.22, 0.22, 0.22])
        # img_np.shape = (224, 224, 3)
        img_np = np.array(image, dtype=float) / 255.
        r = (img_np[:,:,0] - miu[0]) / std[0]
        g = (img_np[:,:,1] - miu[1]) / std[1]
        b = (img_np[:,:,2] - miu[2]) / std[2]
        img_np_t = np.array([r,g,b])
        img_np_nchw = np.expand_dims(img_np_t, axis=0)
        return img_np_nchw
    except:
        print("RuntimeError: get_img_np_nchw({}).".format(filename))
        # NoneType
    

if __name__ == '__main__':
    
    tf_pb_path = 'model_best_checkpoint_resnet18.pth.tar.onnx_graph.pb'
    
    filename = 'pants.jpg'
    img_np_nchw = get_img_np_nchw(filename)
    
    # step 3, check if tf.pb is right.
    with tf.Graph().as_default():
        graph_def = tf.GraphDef()
        with open(tf_pb_path, "rb") as f:
            graph_def.ParseFromString(f.read())
            tf.import_graph_def(graph_def, name="")
        with tf.Session() as sess:
            init = tf.global_variables_initializer()
            #init = tf.initialize_all_variables()
            sess.run(init)
            
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
            output_tf_pb = sess.run([outputs1, outputs2], feed_dict={input_x:img_np_nchw})
            print('output_tf_pb = {}'.format(output_tf_pb))
```

# 3. Related Info 

## 3.1 ONNX

Open Neural Network Exchange
https://github.com/onnx
https://onnx.ai/

The ONNX exporter is a ==**trace-based**== exporter, which means that it operates by executing your model once, and exporting the operators which were actually run during this run. [Limitations](https://pytorch.org/docs/stable/onnx.html#example-end-to-end-alexnet-from-pytorch-to-caffe2)

https://github.com/onnx/tensorflow-onnx
https://github.com/onnx/onnx-tensorflow

## 3.2 Microsoft/MMdnn

当前网络没有调通
https://github.com/Microsoft/MMdnn/blob/master/mmdnn/conversion/pytorch/README.md

# Reference

1. Open Neural Network Exchange https://github.com/onnx
2. [Exporting model from PyTorch to ONNX](https://github.com/onnx/tutorials/blob/master/tutorials/PytorchOnnxExport.ipynb)
3. [Importing ONNX models to Tensorflow(ONNX)](https://github.com/onnx/tutorials/blob/master/tutorials/OnnxTensorflowImport.ipynb)
4. [Tensorflow + tornado服务](https://zhuanlan.zhihu.com/p/26136080)
5. [graph_def = tf.GraphDef() graph_def.ParseFromString(f.read())](https://github.com/llSourcell/tensorflow_image_classifier/blob/master/src/label_image.py)
6. [A Tool Developer's Guide to TensorFlow Model Files](https://www.tensorflow.org/extend/tool_developers/)
7. [TensorFlow学习笔记：Retrain Inception_v3](https://www.jianshu.com/p/613c3b08faea)


