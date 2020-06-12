# Flask demo for pedestrian_hog_python
使用flask为pedestrian_hog_python编写的一个小demo


## Requirements
除pedestrian_hog_python需要的库之外还需要:  
+ Flask >= 1.1.1

## 使用方法
### 配置
使用的配置文件地址和模型地址请更改`app.py`中的`cfg_path`和`model_path`

### 运行
```
python app.py
```

### 在线DEMO（校园网环境）

### 使用流程
1. 建议使用Chromium内核浏览器访问，否则可能会有BUG。正常可以看到如下的页面：

2. 点击左侧**选择**按钮选择图片。建议使用`demo_images`文件夹中的图片，均来自于InriaPerson的测试集（模型训练自InriaPerson的训练集）。

3. 点击左侧**上传**按钮上传图片。

3. 完成图片上传后，点击右侧的**开始预测**按钮，右侧会显示对该图片中是否包含行人的预测结果。

5. *完成图片预测后如想上传新的图片并预测，建议先在左侧预览框里移除旧图片。

### Thanks to
+ [bootstrap 4](https://github.com/twbs/bootstrap)
+ [bootstrap-input](https://github.com/kartik-v/bootstrap-fileinput)