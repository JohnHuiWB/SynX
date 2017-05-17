#!user/bin/env python3
# -*- coding: utf-8 -*-

'''
Created on 20170518
@author: JohnHuiWB
'''

import numpy as np
import operator

class KNN(object):
    def __init__(self):
        self.unknown = 0 # 未知数据的向量 (1xN)
        self.train_data = 0 # M个已知数据的向量 (NxM)
        self.labels = 0 # M个已知数据的标签的向量 (1xM)
        self.k = 5 # 选择前k个最相似的数据 (k一般是不大于20的整数)

    def predict(self, unknown:list, existingdata:list, labels:list, k = 5):
        """
        Input:  unknown: 未知数据的列表 (1xN)
                existingdata: M个已知数据的列表 (NxM)
                labels: M个已知数据的标签的列表 (1xM)
                k: 选择前k个最相似的数据 (k一般是不大于20的整数)
        Output: 算法判断出的未知数据的标签
        """
        self.unknown = np.array(unknown)
        self.train_data = np.array(existingdata)
        self.labels = np.array(labels)
        self.k = k

        data_set_size = self.train_data.shape[0] # 获得已知数据M的大小

        # 用tile函数使unknown扩展为NxM的矩阵，并与已知数据做减法
        diff_mat = np.tile(self.unknown, (data_set_size,1)) - self.train_data
        sq_diff_mat = diff_mat**2 # 平方
        sq_distances = sq_diff_mat.sum(axis=1) # axis=0，表示列，axis=1，表示行。
        distances = sq_distances**0.5 # 开方
        indices = distances.argsort() # 返回从小到大的值的索引

        label_count = {} # 创建计数字典
        for i in range(self.k): # 统计距离最小的前k个标签分别的个数
            label = self.labels[indices[i]]
            label_count[label] = label_count.get(label, 0) + 1
        sorted_label_count = sorted(label_count.items(), key = operator.itemgetter(1), reverse = True)
        # sorted函数返回一个由一个tuple组成的list，如[(0, 3)]

        return sorted_label_count[0][0]