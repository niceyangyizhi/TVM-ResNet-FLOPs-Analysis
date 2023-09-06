# TVM-ResNet-FLOPs-Analysis

FLOPs常用于衡量一个模型的计算量，模型计算量会显著影响模型在边缘侧的部署。这里使用TVM realy前端统计resnet18-v1-7.onnx各类算子的FLOPs。

## Preliminary Work

本实验是基于TVM的，因此需要先安装TVM，TVM的安装指南参见[官方教程](https://tvm.apache.org/docs/install/index.html)

## 实验过程

`count_flops_resnet.py`是程序的入口，会先下载一个resnet18-v1-7.onnx，然后将模型导入TVM，ONNX模型会被转换成realy.Function，其中的node节点会被替换成realy算子。接着调用TVM提供的API `relay.analysis.post_order_visit`遍历生成的realy.Function，在遍历的过程中对遇到的每种算子计算FLOPs，并保存在一个字典中。最后，打印每种算子总共的FLOPs。

`count_flops_op.py`中定义了计算resnet18使用的各类算子FLOPs的函数，涵盖的算子有：
- nn.conv2d
- nn.batch_norm
- nn.relu、add
- nn.max_pool2d
- nn.global_avg_pool2d
- nn.batch_flatten
- nn.dense

算子名称和对应的计算函数保存在一个字典中，以便`count_flops_resnet.py`可以根据算子名调用计算函数。

实验输出结果如下：
```
nn.conv2d 3627122688
nn.batch_norm 9934848
nn.relu 2308096
nn.max_pool2d 0
add 753640
nn.global_avg_pool2d 0
nn.batch_flatten 0
nn.dense 512000
```