#!user/bin/env python3
# -*- coding: utf-8 -*-

'''
Created on 20170911
@author: JohnHuiWB
'''

import os
import sys
sys.path.append("../")
import time
from SynXNLP.classify.SVM import SVM_sklearn_SVC
from SynXNLP.seg.seg import Seg
from SynXNLP.feature.feature_extraction import Extracter


class Test(object):
    """docstring for Main"""
    def __init__(self):
        self.segmenter = Seg()
        self.extracter = Extracter()
        self.estimator = SVM_sklearn_SVC()
        # 初始化默认的测试集路径
        self._data_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.path.pardir)) + '\\SynXNLP\\data\\test'


    def _get_file_list(self):
        """
        解析获取所以文件
        返回文件名的list及每个文件对应的label
        """
        file_list = []
        label_list = []
        for label in os.listdir(self._data_dir):
            for fn in os.listdir(self._data_dir + '\\' + label):
                file_list.append(self._data_dir + '\\' + label + '\\' + fn) # 存为绝对路径
                label_list.append(label)

        return file_list, label_list


    def test_without_cache(self):
        """
        不带缓存，测试系统的性能
        """
        self.extracter.clear_cache() # 清理缓存
        self.estimator.clear_cache() # 清理缓存

        # 开始计时
        train_start = time.clock()

        self.extracter.fit()
        train_data = self.extracter.get_train_vec()
        train_vec = train_data['_train_vec']
        train_labels = train_data['_labels']
        self.estimator.fit(train_vec, train_labels)

        # 结束计时
        train_end = time.clock()
        print('All models are built in ' + str(train_end - train_start) + ' seconds time.')

        file_list, label_list = self._get_file_list()
        
        TP = FP = TN = FN = 0
        # TP：真阳性，是相关项目中被正确识别为相关的。
        # TN：真阴性，是不相关项目中被正确识别为不相关的。
        # FP：假阳性，是不相关项目中被错误被识别为相关的。
        # FN：假阴性，是相关项目中被错误识别为不相关的。

        sum_time = 0

        for i in range(len(label_list)):
            
            predict_start = time.clock() # 开始计时
            result = self.test_analysis_file(file_list[i])
            predict_end = time.clock() # 结束计时

            sum_time += predict_end - predict_start

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
        each_time = sum_time / len(label_list)
        print('平均每个预测需要 ' + str(each_time) + ' 秒.')

        # 计时结束

        print('Test completed.')


    def test_with_cache(self):
        """
        带缓存，测试系统性能
        """
        # train_data = self.extracter.get_train_vec()
        # train_vec = train_data['_train_vec']
        # train_labels = train_data['_labels']
        # self.estimator.fit(train_vec, train_labels)

        file_list, label_list = self._get_file_list()
        
        TP = FP = TN = FN = 0
        # TP：真阳性，是相关项目中被正确识别为相关的。
        # TN：真阴性，是不相关项目中被正确识别为不相关的。
        # FP：假阳性，是不相关项目中被错误被识别为相关的。
        # FN：假阴性，是相关项目中被错误识别为不相关的。

        sum_time = 0

        for i in range(len(label_list)):

            predict_start = time.clock() # 开始计时
            result = self.test_analysis_file(file_list[i])
            predict_end = time.clock() # 结束计时
            
            sum_time += predict_end - predict_start

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
        each_time = sum_time / len(label_list)
        print('平均每个预测需要 ' + str(each_time) + ' 秒.')

        print('Test completed.')


    def test_analysis_file(self, filename):
        with open(filename, 'r', encoding = 'utf8') as fp:
            test_text = fp.read()
        seg_text = self.segmenter(test_text)
        vec = self.extracter([seg_text])
        return self.estimator(vec)[0]


    def _cal_performance(self):
        pass
        

def main():
    # T = Test()
    # T.test_with_cache()


if __name__ == '__main__':
    s = time.clock()
    main()
    e = time.clock()
    print('Finished in ' + str(e-s) + ' s.')