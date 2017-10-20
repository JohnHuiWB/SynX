#!user/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on 20170819
@author: JohnHuiWB
"""

import random
import numpy


class Simple_SVM(object):
    """docstring for Simple_SVM"""
    def __init__(self):
        self._b = 0 # 初始化为0
        self._alphas = 0 # 初始化为0 
        self._w = 0 # 初始化为0 


    def _select_j_rand(self, i, m):
        """
        随机选择j，j不等于i
        Input:  i 第一个alpha的下标
                m 所有alpha的数目
        Output: j
        """
        j = i
        while j == i:
            j = int(random.uniform(0, m))
        return j


    def _clip_Alpha_j(self, alpha_j, H, L):
        """
        调整大于H或小于L的alpha的值
        Input:  alpha
                H alpha的上限
                L alpha的下限
        Output: alpha
        """
        if alpha_j > H:
            alpha_j = H
        elif L > alpha_j:
            alpha_j = L
        return alpha_j


    def smo_simple(self, data_set, labels, C, tolerance, max_iter):
        """ 
        简化版smo算法
        Input:  data_set 数据集
                labels 类别标签
                C 离群点的权重
                tolerance 松弛变量常数
                max_iter 退出前最大的循环次数 
        Output: self._b
                self._alphas
        """
        data_matrix = numpy.mat(data_set) # 转换为m*n的矩阵
        m, n = numpy.shape(data_matrix)
        labels_matrix = numpy.mat(labels).transpose() # 转换为m*1的矩阵
        self._b = 0 # 重新设置为0
        self._alphas = numpy.mat(numpy.zeros((m, 1))) # 初始化alphas为m*1的全零矩阵
        cur_iter = 0 # 当前循环的次数

        while cur_iter < max_iter:
            alpha_pairs_changed = 0 # 用于记录alpha是否已经进行优化

            for i in range(m):
                forecast_i = numpy.multiply(self._alphas, labels_matrix).T * (data_matrix * data_matrix[i,:].T) + self._b # 用当前模型预测的类别
                Ei = forecast_i - labels_matrix[i] # 预测结果与实际结果的误差
                
                if ((labels_matrix[i] * Ei < -tolerance) and self._alphas[i] < C) or ((labels_matrix[i] * Ei > tolerance) and self._alphas[i] > 0):
                    # i的函数间隔数值超过松弛变量的限制，但其对应的alpha值不等于C时
                    # 或者 i的函数间隔在容忍范围内，但其对应的alpha值不等于0时
                    # 在这两种情况下，需要更新alpha的值，优化超平面
                    j = self._select_j_rand(i, m) # 随机选择一个j，j的值不能于i的值相等
                    forecast_j = numpy.multiply(self._alphas, labels_matrix).T * (data_matrix * data_matrix[j,:].T) + self._b # 用当前模型预测的类别
                    Ej = forecast_j - labels_matrix[j] # 预测结果与实际结果的误差
                    alpha_i_old = self._alphas[i].copy()
                    alpha_j_old = self._alphas[j].copy()
                    # 保证新的alpha[j]在0和C之间
                    if labels_matrix[i] != labels_matrix[j]:
                        L = max(0, self._alphas[j] - self._alphas[i])
                        H = min(C, C + self._alphas[j] - self._alphas[i])
                    else:
                        L = max(0, self._alphas[j] + self._alphas[i] - C)
                        H = min(C, self._alphas[j] + self._alphas[i])
                    if L == H:
                        print('L == H')
                        continue

                    eta = 2.0 * data_matrix[i,:] * data_matrix[j,:].T - data_matrix[i,:] * data_matrix[i,:].T - data_matrix[j,:] * data_matrix[j,:].T
                    if eta >= 0:
                        print('eta >= 0')
                        continue

                    self._alphas[j] -= labels_matrix[j] * (Ei - Ej) / eta
                    self._alphas[j] = self._clip_Alpha_j(self._alphas[j], H, L)

                    if abs(self._alphas[j] - alpha_j_old) < 0.00001:
                        print('j not moving enough')
                        continue

                    self._alphas[i] += labels_matrix[j] * labels_matrix[i] * (alpha_j_old - self._alphas[j])

                    b1 = self._b - Ei - labels_matrix[i]*(self._alphas[i]-alpha_i_old)*data_matrix[i,:]*data_matrix[i,:].T - labels_matrix[j]*(self._alphas[j]-alpha_j_old)*data_matrix[i,:]*data_matrix[j,:].T
                    b2 = self._b - Ej - labels_matrix[i]*(self._alphas[i]-alpha_i_old)*data_matrix[i,:]*data_matrix[j,:].T - labels_matrix[j]*(self._alphas[j]-alpha_j_old)*data_matrix[j,:]*data_matrix[j,:].T
                    if 0 < self._alphas[i] and self._alphas[i] < C:
                        self._b = b1
                    elif 0 < self._alphas[j] and self._alphas[j] < C:
                        self._b = b2
                    else:
                        self._b = (b1 + b2) / 2
                    
                    alpha_pairs_changed += 1
                    print('iter: %d i: %d, pairs changed: %d' % (cur_iter, i, alpha_pairs_changed))
            if alpha_pairs_changed == 0:
                cur_iter += 1
            else:
                cur_iter = 0
            print('iteration number: %d' % cur_iter)

        self._w = numpy.multiply(self._alphas, labels_matrix).T * data_matrix
        print(self._w)
        
        return self._b, self._alphas


from sklearn.svm import SVC
import os
import sys
from sklearn.externals import joblib


class SVM_sklearn_SVC(object):
    """
    调用sklearn的SVC模块
    实现SVM的功能
    """
    def __init__(self, extracter):
        # 初始化已经缓存的模型的路径
        self._model_path = os.path.join(os.path.dirname(os.path.abspath(__file__))) + '\\SVM.model'
        self._clf = None # 初始化
        self.extracter = extracter

        if os.path.exists(self._model_path) is True:
            self._load() # 加载
        else:
            self._clf = SVC(C = 1.3, tol = 0.0005, probability = True)
            self._fit()


    def _fit(self):
        """
        调用fit，然后保存新的model
        """
        print('Building new SVM model ...')
        train_data = self.extracter.get_train_vec()
        train_vec = train_data['_train_vec']
        train_labels = train_data['_labels']
        self._clf.fit(train_vec, train_labels)
        self._save()

    
    def _save(self):
        """
        保存model
        """
        print('Saving the new SVM model to the path ' + self._model_path)
        joblib.dump(self._clf, self._model_path)


    def _load(self):
        """
        加载model
        """
        print('Loading SVM model from the path ' + self._model_path)
        self._clf = joblib.load(self._model_path)


    def __call__(self, X):
        """
        调用predict_proba方法
        """
        return self._clf.predict_proba(X)


    def clear_cache(self):
        """
        手动清理缓存
        """
        if os.path.exists(self._model_path) is True:
            os.remove(self._model_path)



