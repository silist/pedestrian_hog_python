# -*- encoding: utf-8 -*-

"""
@File    :   utils.py
@Time    :   2020/06/03 15:44:40
@Author  :   silist
@Version :   1.0
@Desc    :   用于计算HOG的辅助方法
"""

import numpy as np
from scipy.signal import convolve2d


def get_gradient_operator(method='simple'):
    """根据不同方法得到相应x和y方向的梯度算子

    Arguments:
        method {str} -- 'simple' or 'sobel'. 默认为'simple'

    Returns:
        tuple(np.ndarray, np.ndarray) -- x和y方向的梯度算子
    """
    if method == 'simple':
        # 注意convolve2d输入算子必须为2维
        op = np.array([[-1, 0, 1]])
        return op, op.T
    elif method == 'sobel':
        sobel_x_operator = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        sobel_y_operator = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
        return sobel_x_operator, sobel_y_operator

def calculate_gradient_xy(img, operator_x, operator_y):
    """通过卷积得到x和y方向梯度"""
    g_x = convolve2d(img, operator_x, mode="same", boundary="symm")
    g_y = convolve2d(img, operator_y, mode="same", boundary="symm")
    return g_x, g_y

def calculate_magnitude(g_x, g_y):
    """通过x和y方向梯度得到对应梯度幅值"""
    g = np.sqrt(np.power(g_x, 2) + np.power(g_y, 2))
    return g

def calculate_orientation(g_x, g_y):
    """通过x和y方向梯度得到对应梯度方向，值域[-pi/2, pi/2]"""
    theta = np.arctan(g_y / (g_x + np.finfo(float).eps))
    return theta

def vector_normalization(vector, method):
    """对于输入向量归一化

    Args:
        vector (np.ndarry): 输入向量
        method (str): 'L1-norm', 'L1-sqrt', 'L2-norm' or 'L2-Hys'

    Returns:
        [np.ndarray]: 归一化后的向量
    """    
    vec_norm = np.zeros(vector.shape)
    eps = np.finfo(float).eps
    if method == 'L1-norm':
        vec_norm = vector / (abs(vector).sum() + eps)
    elif method == 'L1-sqrt':
        vec_norm = vector / np.sqrt(abs(vector).sum() + eps)
    elif method == 'L2-norm':
        vec_norm = vector / np.sqrt(np.power(vector, 2).sum() + eps)
    elif method == 'L2-Hys':
        vec_norm = vector / np.sqrt(np.power(vector, 2).sum() + eps)
        vec_norm = np.minimum(vec_norm, 0.2)    # 以0.2截断
        vec_norm /= np.sqrt(np.power(vec_norm, 2).sum() + eps)
    return vec_norm
