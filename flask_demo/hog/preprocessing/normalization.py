# -*- encoding: utf-8 -*-

"""
@File    :   normalization.py
@Time    :   2020/06/01 16:07:53
@Author  :   silist
@Version :   1.0
@Desc    :   对灰度图进行标准化。
             1. Gamma矫正
             2. Gaussian平滑
"""

import math
import numpy as np
from scipy.signal import convolve2d

def gamma_correction(img, gamma):
    '''gamma: float, 用于伽马矫正的gamma值'''
    return np.power(img, gamma)

def gaussian_smoothing(img, shape=(5, 5), sigma=0.5):
    def gaussian_kernel():
        m, n = [(ss-1.)/2. for ss in shape]
        y, x = np.ogrid[-m: m+1, -n: n+1]
        h = 1 / (math.sqrt(2*math.pi)*sigma) * np.exp( -(x*x + y*y) / (2.*sigma*sigma) )
        h[h < np.finfo(h.dtype).eps*h.max()] = 0
        return h
    return convolve2d(img, gaussian_kernel(), mode="same", boundary="symm")
