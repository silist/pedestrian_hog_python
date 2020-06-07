# -*- encoding: utf-8 -*-

"""
@File    :   inria_loader.py
@Time    :   2020/05/30 17:05:26
@Author  :   silist
@Version :   1.0
@Desc    :   Data loader for Inria Person dataset.
"""

import os
import numpy as np
from skimage import io, transform

class InriaDataLoader(object):
    def __init__(self, cfg, dtype='train'):
        """通过配置文件加载Inria数据集

        Arguments:
            cfg {dict} -- cfg['dataset']

        Keyword Arguments:
            dtype {str} -- train or test. (default: {'train'})
        """
        self.cfg = cfg
        self.dataset = self._load_dataset(cfg, dtype)
        self.type = dtype

        
    def _load_dataset(self, cfg, dtype):
        # 加载正负样本并生成对应标签: 正-1，负-0
        def load_img_from_path(path, as_gray=True):
            """默认加载为灰度图"""
            img = io.imread(path, as_gray=True)
            if 'resize' in cfg and cfg['resize']:
                img = self._resize(img, cfg['resize'])
            return img
        pos_img_path = cfg['%s_pos_img_path' % dtype]
        all_pos_img_paths = [os.path.join(pos_img_path, p) for p in os.listdir(pos_img_path)]
        neg_img_path = cfg['%s_neg_img_path' % dtype]
        all_neg_img_paths = [os.path.join(neg_img_path, p) for p in os.listdir(neg_img_path)]
        pos_images = [load_img_from_path(p) for p in all_pos_img_paths]
        neg_images = [load_img_from_path(p) for p in all_neg_img_paths]
        pos_labels = np.ones(len(pos_images), dtype=int)
        neg_labels = np.zeros(len(neg_images), dtype=int)
        # 合并为完整数据集并打乱
        images = np.array(pos_images + neg_images)
        labels = np.append(pos_labels, neg_labels)
        all_img_paths = np.array(all_pos_img_paths + all_neg_img_paths)
        if 'shuffle' in cfg and cfg['shuffle']:
            # print(images.shape, labels.shape)
            idx = np.random.permutation(images.shape[0])
            images, labels = images[idx], labels[idx]
            all_img_paths = all_img_paths[idx]
        self.img_paths = all_img_paths
        self.size = len(all_img_paths)
        return images, labels

    def _resize(self, img, output_shape):
        return transform.resize(img, output_shape)

    def __getitem__(self, index):
        """内置函数，每次调用时返回index对应的图片及标签
        """
        return (self.dataset[0][index], self.dataset[1][index])
    
    def __len__(self):
        return self.size
