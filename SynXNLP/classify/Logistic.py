#!user/bin/env python3
# -*- coding: utf-8 -*-

'''
Created on 20170518
@author: JohnHuiWB
'''

import numpy as np
import random

class Logistic(object):
    def __init__(self):
        self._weights = [] # 所求权值

    def train(self, train_data:list, labels:list, iter = 150, empirical_parameter = 0.01):
        """
        采用随机梯度上升法，进行logistic回归
        Input:  train_data 要求为标准的数值型数据集
                labels 数据集对应的类别
                iter 对整个数据集的迭代次数，默认为200次
                empirical_parameter 经验参数，默认为0.01
        Output: 
        """
        train_data = np.mat(train_data) # 转换为 数据个数*自变量个数 的矩阵
        labels = np.mat(labels).T # 转换为列矩阵，便于计算
        m, n = np.shape(train_data)
        self._weights = np.ones((n, 1))
        print("Building model......")
        for j in range(iter):
            data_indx = list(range(m))
            for i in range(m):
                # 使alpha随着迭代次数不断减小
                alpha = 4 / (1.0 + i + j) + empirical_parameter 
                rand_indx = random.choice(data_indx) # 随机选取更新

                h = self._sigmiod(np.sum(train_data[rand_indx] * self._weights))
                error = labels[rand_indx] - h;
                self._weights += alpha * train_data[rand_indx].T * error
                data_indx.remove(rand_indx)
        print('Finish')

        return self._weights

    def _sigmiod(self, x):
        return 1.0 / (1 + np.exp(-x))

    def train_result(self):
        return self._weights

    def predict(self, data:list):
        data = np.mat(data)
        prob = self._sigmiod(np.sum(data * self._weights))

        if prob > 0.5: return 1
        else: return 0