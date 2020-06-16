# -*- encoding: utf-8 -*-

"""
@File    :   utils.py
@Time    :   2020/06/11 15:43:34
@Author  :   silist
@Version :   1.0
@Desc    :   Utils for flask demo.
"""

import pickle
import numpy as np
from skimage import io, transform

from hog.loader.config_loader import Config
from hog.loader.inria_loader import InriaDataLoader
from hog.preprocessing.normalization import gamma_correction
from hog.feature_extraction.hog import RHOG

def get_feature(cfg, hog, img_path):
    """读取输入图片，标准化并提取特征"""
    img = io.imread(img_path, as_gray=True)
    if 'resize' in cfg['dataset'] and cfg['dataset']['resize']:
        img = transform.resize(img, cfg['dataset']['resize'])
    img = gamma_correction(img, cfg['normalization']['gamma'])
    feature_vector = hog.get_img_hog(img)
    feature_vector = feature_vector.reshape(1, -1)   # 2D-array for sklearn
    return feature_vector

def get_hog(cfg):
    return RHOG(cfg['hog'])

def get_prediction(model, feature):
    return int(model.predict(feature))

def load_model(model_path):
    """加载SVM模型"""
    return pickle.load(open(model_path, 'rb'))
