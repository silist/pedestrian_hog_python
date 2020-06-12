# -*- encoding: utf-8 -*-

"""
@File    :   train.py
@Time    :   2020/06/01 16:11:10
@Author  :   silist
@Version :   1.0
@Desc    :   根据配置文件训练SVM模型并保存
"""

import pickle
import numpy as np

from tqdm import tqdm

from loader.config_loader import Config
from loader.inria_loader import InriaDataLoader
from preprocessing.normalization import gamma_correction
from feature_extraction.hog import RHOG
from classifier.svm import get_svm

if __name__ == "__main__":
    cfg_path = r'./config/hog_svm_inria_pc.yml'
    cfg = Config(cfg_path)
    print('Config:\n', cfg.cfg)
    # 根据配置文件构建训练数据集
    print('Loading dataset...')
    dataloader = InriaDataLoader(cfg['dataset'], dtype='train')
    print('Dataset size: %s' % len(dataloader))
    # 构建HOG
    hog = RHOG(cfg['hog'])
    # 构造特征矩阵和标签矩阵
    print('Building feature matrix...')
    rog_features, labels = [], []
    # 对于每张图片标准化并提取特征
    for img, label in tqdm(dataloader):
        img = gamma_correction(img, cfg['normalization']['gamma'])
        feature_vector = hog.get_img_hog(img)
        rog_features.append(feature_vector)
        labels.append(label)
    rog_features = np.array(rog_features)
    labels = np.array(labels)
    # 输入SVM进行训练
    print('Training model...')
    svm = get_svm(cfg['svm'])
    svm.fit(rog_features, labels)
    # 保存模型
    model_save_path = r'./pkl/hog_inria_svm_linear_interp_none_flask.pkl'
    print('Saving model to %s...' % model_save_path)
    pickle.dump(svm, open(model_save_path, 'wb'))
    print('Finish~')
