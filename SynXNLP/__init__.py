#!user/bin/env python3
# -*- coding: utf-8 -*-

'''
Created on 20170911
@author: JohnHuiWB
'''

import os

from .classify.SVM import SVM_sklearn_SVC
from .seg.seg import Seg
from .feature.feature_extraction import Extracter
from .crawler.Webcrawler import Webcrawler



class Filter(object):
    """docstring for Filter"""
    def __init__(self):
        self.segmenter = Seg()
        self.extracter = Extracter()
        self.estimator = SVM_sklearn_SVC(self.extracter)
        self.wc = Webcrawler()
        self.filename = 'E:\\repository\\SynX-NLP\\cache.txt'
        self.text = None
        self.seg_text = None
        self.vec = None
        self.result = None


    def analysis(self):
        if os.path.exists(self.filename) is False:
            print('Please download a file first.')
            return -2
        with open(self.filename, 'r', encoding = 'utf8') as fp:
            self.text = fp.read()
        self.seg_text = self.segmenter(self.text)
        self.vec = self.extracter([self.seg_text])
        self.result = self.estimator(self.vec)[0]
        if self.result[1] < 0.8:
            return -1
        else:
            return 1

    def get_data(self, url:str):
        data = self.wc(url)
        with open(self.filename, 'w', encoding = 'utf8') as fp:
            fp.writelines(data)
    

    def print_text(self):
        if self.filename is '':
            print('Please download a file first.')
            return -2
        with open(self.filename, 'r', encoding = 'utf8') as fp:
            self.text = fp.read()
        print(self.text)


    def print_seg_text(self):
        if self.seg_text is None:
            print('Please run \'analysis\' method first.')
            return -2
        print(self.seg_text)


    def print_vec(self):
        if self.vec is None:
            print('Please run \'analysis\' method first.')
            return -2
        print(self.vec)


    def print_result(self):
        if self.result is None:
            print('Please run \'analysis\' method first.')
            return -2
        print(self.result)


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
get_data = F.get_data
print_text = F.print_text
print_seg_text = F.print_seg_text
print_vec = F.print_vec
print_result = F.print_result
