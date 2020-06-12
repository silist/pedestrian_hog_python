# -*- encoding: utf-8 -*-

"""
@File    :   test.py
@Time    :   2020/06/04 16:14:47
@Author  :   silist
@Version :   1.0
@Desc    :   在测试集上测试训练好的SVM模型
"""

import pickle
import numpy as np

from tqdm import tqdm
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score

from loader.config_loader import Config
from loader.inria_loader import InriaDataLoader
from preprocessing.normalization import gamma_correction
from feature_extraction.hog import RHOG

if __name__ == "__main__":
    cfg_path = r'./config/hog_svm_inria_pc.yml'
    cfg = Config(cfg_path)
    print('Config:\n', cfg.cfg)
    # 根据配置文件构建训练数据集
    print('Loading dataset...')
    dataloader = InriaDataLoader(cfg['dataset'], dtype='test')
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
    # 加载训练好的SVM模型
    model_fp = r'./pkl/hog_inria_svm_linear_interp_none_flask.pkl'
    print('Loading model...')
    svm = pickle.load(open(model_fp, 'rb'))
    labels_pred = svm.predict(rog_features)
    print('Metrics:')
    print('Accuracy: %.5f' % accuracy_score(labels, labels_pred))
    print('Recall: %.5f' % recall_score(labels, labels_pred))
    print('Precision: %.5f' % precision_score(labels, labels_pred))
    print('F1-score: %.5f' % f1_score(labels, labels_pred))
