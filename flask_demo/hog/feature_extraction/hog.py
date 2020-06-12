# -*- encoding: utf-8 -*-

"""
@File    :   hog.py
@Time    :   2020/06/03 14:56:59
@Author  :   silist
@Version :   1.0
@Desc    :   使用R-HOG做特征提取
"""

import numpy as np
from .utils import *


class RHOG():
    def __init__(self, cfg):
        self.cfg = cfg
        # self.partition_bins(num_bins)
    
    def _get_gradient(self, img, gradient_operator):
        """返回梯度幅值和方向"""
        op_x, op_y = get_gradient_operator(gradient_operator)
        g_x, g_y = calculate_gradient_xy(img, op_x, op_y)
        magnitude = calculate_magnitude(g_x, g_y)
        orientation = calculate_orientation(g_x, g_y)
        # from skimage.io import imsave
        # imsave('images/img.png', img)
        # imsave('images/g_x.png', g_x)
        # imsave('images/g_y.png', g_y)
        # imsave('images/magnitude.png', magnitude)
        # imsave('images/orientation.png', orientation)
        return magnitude, orientation

    def _calculate_cell_hogs(self, magnitude, orientation, bins, 
                            cell_size, interpolation):
        """计算每个单元的hog并整合，返回矩阵shape=(m_cells*bins, n_cells)"""
        m, n = magnitude.shape
        m_cells, n_cells = m // cell_size, n // cell_size
        cell_slices = []    # 对每个cell的划分
        for i in range(m_cells):
            for j in range(n_cells):
                # 从图片左上角开始根据cell_size划分cell，多余部分直接舍去
                cell_slices.append(np.s_[i*cell_size: (i+1)*cell_size, j*cell_size: (j+1)*cell_size])
        # 对[-2/pi, 2/pi]范围均分为bins份
        bins_slices = np.linspace(-np.pi/2, np.pi/2, bins+1)
        def get_bins_num(theta):
            for s in range(len(bins_slices) - 1):
                if bins_slices[s] <= theta < bins_slices[s+1]:
                    return s
            if theta == bins_slices[-1]:
                    return len(bins_slices) - 1
            else:
                raise OverflowError("Theta must be in [-pi/2, pi/2], theta=%.3f" % theta)
        # all_cell_hogs = np.zeros(shape=(m_cells * bins, n_cells))
        all_cell_hogs = np.empty(shape=(m_cells, n_cells, bins))
        # TODO 实现三线性插值
        if interpolation == 'trilinear':
            # 根据x, y, theta构造三维矩阵，根据每个cube的中点坐标插值
            raise NotImplementedError("Trilinear interpolation is not implemented!")
        elif interpolation == 'none':
            func_vec = np.vectorize(get_bins_num)
            ori_bin_nums = func_vec(orientation)
            for s in cell_slices:
                # 每个cell根据角度范围投票至相应bin
                cell_hog_votes = np.zeros(shape=(bins, ))
                mag_cell, ori_cell = magnitude[s], ori_bin_nums[s]
                for i in range(bins):
                    cell_hog_votes[i] = mag_cell[np.where(ori_cell == i)].sum()
                # 将cell_hog_votes根据cell位置放入all_cell_hogs
                cell_i, cell_j = s[0].start // cell_size, s[1].start // cell_size
                # all_cell_hogs[cell_i: cell_i+bins, cell_j] = cell_hog_votes
                all_cell_hogs[cell_i, cell_j] = cell_hog_votes
            return all_cell_hogs

    def _calculate_block_hogs(self, cell_hogs, block_size, 
                            block_stride, norm_method):
        """计算每个block的rog并整合"""
        m_cells, n_cells, bins = cell_hogs.shape
        # m_cells //= bins
        block_slices = []   # 对每个block的划分，考虑overlap
        for i in range(0, m_cells - block_size + 1, block_stride):
            for j in range(0, n_cells - block_size + 1, block_stride):
                block_slices.append(np.s_[i: i+block_size, j: j+block_size])
        result_rogs = []
        for s in block_slices:
            block_rog_vec = cell_hogs[s].flatten()
            block_rog_vec = vector_normalization(block_rog_vec, norm_method)
            result_rogs.append(block_rog_vec)
        result_rogs = np.array(result_rogs)
        return result_rogs

    def get_img_hog(self, img):
        """对输入图片返回对应配置参数的HOG特征向量"""
        magnitude, orientation = self._get_gradient(img, self.cfg['gradient_operator'])
        cell_hogs = self._calculate_cell_hogs(magnitude, orientation, self.cfg['bins'], self.cfg['cell_size'], self.cfg['interpolation'])
        block_hogs = self._calculate_block_hogs(cell_hogs, self.cfg['block_size'], self.cfg['block_stride'], self.cfg['norm_method'])
        return block_hogs.flatten()
