# Pedestrian detection with HOG and SVM

参考论文：[Histograms of Oriented Gradients for Human Detection](http://www2.cs.duke.edu/courses/spring19/compsci527/papers/Dalal.pdf)  
基于python3实现了HOG算法，并使用SVM在InriaPerson数据集上训练得到模型。

## Requirements
+ scipy >= 1.0.0
+ numpy >= 1.13.3
+ scikit-learn >= 0.19.1
+ scikit-image >= 0.15.0
+ PyYAML >= 3.13
+ tqdm >= 4.0.0

## 使用方法
### Config
配置文件采用YAML格式，在`config/hog_svm_inria.yml`中给出了一个样例。具体说明如下：
#### dataset:
+ train_pos_img_path: 用于训练的图片正例文件夹地址，InriaPerson数据集已经完成了这部分的分割，直接使用即可。
+ train_neg_img_path: 用于训练的图片负例文件夹地址。
+ test_pos_img_path: 用于测试的图片正例文件夹地址。
+ test_neg_img_path: 用于测试的图片负例文件夹地址。
+ shuffle: bool, 可选，是否打乱输入图片。
+ resize: list, 改变输入图片大小。
#### normalization(可选):
+ gamma: 可选，用于输入图片Gamma矫正的gamma值。
#### hog
+ gradient_operator: simple or sobel, 用于计算梯度的算子，simple为[-1, 0, 1]、sobel为sobel算子。
+ bins: int, hog划分角度的份数。
+ interpolation: `none/trilinear`，hog统计bins时使用的插值方法。none为不插值。trilinear三线性插值目前未实现。
+ cell_size: int, hog划分cell的大小（正方形），单位pixel。
+ block_size: int, hog划分block的大小（正方形），单位cell。
+ block_stride: int, block滑动时的步长，单位cell。
+ norm_method: `L1-norm/L1-sqrt/L2-norm/L2-Hys`，用于对block得到的hog向量归一化的方法。具体请参照原论文。
#### svm
+ kernel: 即sklearn.SVM中的核函数选项。

### Train
在`train.py`文件中提供了完整的训练流程，使用时请注意配置文件地址和模型保存地址。

### Test
在`test.py`文件中提供了完整的测试流程，使用时请注意配置文件地址和加载模型地址。

## DEMO
提供的flask demo请查看`flask_demo`文件夹中内容。`flask_demo/README.md`中提供了对该demo的介绍。  
在线运行请访问（校园网环境）：


## TODO
1. 在`feature_extraction/hog.py`中实现三线性插值算法
