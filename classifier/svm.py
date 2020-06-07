# -*- encoding: utf-8 -*-

"""
@File    :   svm.py
@Time    :   2020/06/05 10:36:05
@Author  :   silist
@Version :   1.0
@Desc    :   根据config简单对sklearn.svm进行封装
"""

from sklearn.svm import SVC

class SVM(SVC):
    def __init__(self, cfg):
        if cfg['kernel'] == 'HIK':
            # TODO 实现HIKSVM
            raise NotImplementedError("HIKSVM is not implemented!")
        else:
            super(SVM, self).__init__(kernel=cfg['kernel'])
    
    def __str__(self):
        return super().__str__()
