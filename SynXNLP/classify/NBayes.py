#!user/bin/env python3
# -*- coding: utf-8 -*-

'''
Created on 20170515
@author: JohnHuiWB
'''

import numpy as np

class NBayes(object):
    def __init__(self):
        self._Pyi = {} # p(yi) 是一个词典
        self._datalen = -1 # 训练集的长度
        self._diclen = -1 # 字典长度
        self._labels = [] # 对应每个文本的分类，是一个从外部导入的列表
        self._train_data = [] # 训练数据集
        self._tdm = 0 # p(x|yi)


    def train(self, train_data:list, labels:list):
        # 数据集与类别不对应
        if len(train_data) != len(labels):
            raise ValueError("training data and labels must have same length.")

        self._datalen = len(train_data)
        self._diclen = len(train_data[0])
        self._labels = labels
        self._train_data = np.array(train_data)
        self.cal_Pyi() # 计算在数据集中每个分类的概率：P(yi)
        self.cal_tdm() # 按分类累计向量空间的每维值：P(x|yi) 


    # 计算在数据集中每个分类的概率：P(yi) 
    def cal_Pyi(self):
        label_kinds = set(self._labels) # 获取全部分类
        for x in label_kinds:   
            self._Pyi[x] = self._labels.count(x) / self._datalen


    # 按分类累计向量空间的每维值：P(x|yi)
    def cal_tdm(self):
        self._tdm = np.zeros([len(self._Pyi), self._diclen]) # 创建 类别*字典长度 的零矩阵
        sumlist = np.zeros([len(self._Pyi), 1]) # 统计每个分类的总值
        
        for indx in range(self._datalen):
            self._tdm[self._labels[indx]] += self._train_data[indx] # 将同一类别的词向量空间值加总
            sumlist[self._labels[indx]] = np.sum(self._tdm[self._labels[indx]]) # 统计每个分类的总值————是一个向量
        self._tdm = self._tdm / sumlist # P(x|yi)


    def predict(self, input_data):
    # 预测分类结果，输出预测的分类类别
        test_data = np.array(input_data)
        if np.shape(test_data)[0] != self._diclen:
            # 测试集长度与词典长度不相等
            raise ValueError("testdata and dicLen must have same length.")

        predict_value = 0 # 初始化类别概率
        predict_label = '' # 初始化类别名称
        for tdm_vec, Pyi in zip(self._tdm, self._Pyi):
        # P(x|yi)  P(yi)
            temp = np.sum(test_data * tdm_vec * self._Pyi[Pyi])
            if temp > predict_value:
                predict_value, predict_label = temp, Pyi
        return predict_label