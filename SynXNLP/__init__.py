#!user/bin/env python3
# -*- coding: utf-8 -*-

'''
Created on 20170911
@author: JohnHuiWB
'''

from .classify.SVM import SVM_sklearn_SVC
from .seg.seg import Seg
from .feature.feature_extraction import Extracter



class Filter(object):
    """docstring for Filter"""
    def __init__(self):
        self.segmenter = Seg()
        self.extracter = Extracter()
        self.estimator = SVM_sklearn_SVC(self.extracter)
        # 初始化默认的测试集路径


    def analysis(self, filename):
        with open(filename, 'r', encoding = 'utf8') as fp:
            test_text = fp.read()
        seg_text = self.segmenter(test_text)
        vec = self.extracter([seg_text])
        result = self.estimator(vec)[0]
        if result[1] < 0.8:
            return -1
        else:
            return 1

    def test(self, data_dir):
        """
        测试系统的性能
        """
        import os

        file_list = []
        label_list = []
        for label in os.listdir(data_dir):
            for fn in os.listdir(data_dir + '\\' + label):
                file_list.append(data_dir + '\\' + label + '\\' + fn) # 存为绝对路径
                label_list.append(label)

        TP = FP = TN = FN = 0
        # TP：真阳性，是相关项目中被正确识别为相关的。
        # TN：真阴性，是不相关项目中被正确识别为不相关的。
        # FP：假阳性，是不相关项目中被错误被识别为相关的。
        # FN：假阴性，是相关项目中被错误识别为不相关的。

        for i in range(len(label_list)):

            with open(file_list[i], 'r', encoding = 'utf8') as fp:
                test_text = fp.read()
            seg_text = self.segmenter(test_text)
            vec = self.extracter([seg_text])
            result = self.estimator(vec)[0]

            if int(label_list[i]) == 1:
                if 0.8 < result[1]:
                    print('NO.%3d被正确识别 probability:%s filename:%s' % (i+1, result, file_list[i]))
                    TP += 1
                else:
                    print('NO.%3d被错误识别 probability:%s filename:%s' % (i+1, result, file_list[i]))
                    FN += 1
            else:
                if 0.8 > result[1]:
                    print('NO.%3d被正确识别 probability:%s filename:%s' % (i+1, result, file_list[i]))
                    TN += 1
                else:
                    print('NO.%3d被错误识别 probability:%s filename:%s' % (i+1, result, file_list[i]))
                    FP += 1
        print('TP:%4d FN:%4d TN:%4d FP:%4d' % (TP, FN, TN, FP))
        P = TP / (TP+FP)
        print('准确率：' + str(P))
        R = TP / (TP+FN)
        print('召回率：' + str(R))
        F_measure = 2*P*R/(P+R)
        print('F-measure：' + str(F_measure))

        print('Test completed.')


# default Filter instance

F = Filter()

# global functions

analysis = F.analysis
test = F.test
