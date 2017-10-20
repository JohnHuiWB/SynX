#!user/bin/env python3
# -*- coding: utf-8 -*-

'''
Created on 20170910
@author: JohnHuiWB
@author: Zevan
'''

import os
import sys
import pickle
import codecs
from sklearn.externals import joblib
from sklearn.feature_extraction.text import TfidfVectorizer


class Extracter(object):
    """
    传入一段文本
    进行特征提取
    返回结果
    """
    def __init__(self):
        """
        data_dir：训练集的绝对路径
        """
        # 初始化已经缓存的模型的路径
        self._model_path = os.path.join(os.path.dirname(os.path.abspath(__file__))) + '\\vectorizer.model'
        # 初始化已经缓存的train_vec的路径
        self._train_vec_path = os.path.join(os.path.dirname(os.path.abspath(__file__))) + '\\train_vec.cache'
        self._vectorizer = None # 初始化
        self._train_vec = None # 初始化
        self._data_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.path.pardir)) + '\\data\\train' # 初始化

        if os.path.exists(self._model_path) is False:
            # 初始化默认的训练集路径       
            if os.path.exists(self._data_dir) is False:
                print('训练集不存在!!!')
                exit(1);
            else:
                self.fit() # 训练
        else:
            self._load_model() # 加载


    def __call__(self, test_text:str):
        """
        传入一段文本
        进行特征提取
        返回结果
        """
        # 得到tfidf的矩阵
        tfidf_test = self._vectorizer.transform(test_text)
        return tfidf_test.toarray()


    def _save_model(self):
        """
        保存model和train_vec
        """
        print('Saving the new vectorizer model to the path ' + self._model_path)
        joblib.dump(self._vectorizer, self._model_path)
    

    def _save_vec(self):
        with open(self._train_vec_path, 'wb') as fp:
            print('Saving the new train vec to the path ' + self._train_vec_path)
            pickle.dump(self._train_vec, fp)


    def _load_model(self):
        """
        加载model和train_vec
        """
        print('Loading vectorizer model from the path ' + self._model_path)
        self._vectorizer = joblib.load(self._model_path)
        

    def _load_vec(self):
        with open(self._train_vec_path, 'rb') as fp:
            print('Loading train vec from the path ' + self._train_vec_path)
            self._train_vec = pickle.load(fp)


    def fit(self):
        """
        根据训练集，训练model
        """
        print('Building new vectorizer model from the path ' + self._data_dir)
        # 获取文件列表
        file_list, label_list = self._get_file_list()
        # 加载训练集数据
        train = self._get_train_data(file_list)
        # 创建新的vectorizer
        self._vectorizer = TfidfVectorizer()
        # 用train数据来fit
        vec = self._vectorizer.fit_transform(train)
        # 转换为array
        vec = vec.toarray()
        # 存储为字典类型
        self._train_vec = {'_train_vec':vec, '_labels':label_list}
        # 保存
        self._save_model()
        self._save_vec()
        

    def _get_train_data(self, file_list):
        """
        根据文件列表，通过Seg解析每个文件
        返回train数据
        """
        # 添加上一层的path
        sys.path.append(os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.path.pardir))) 
        from seg.seg import Seg
        S = Seg()

        train = []
        for fn in file_list:
            with open(fn, 'r', encoding='utf8') as fp:
                text = fp.read()
                if text[:3] == codecs.BOM_UTF8:
                    text = text[3:]

            # print('Segmenting ' + fn)
            train.append(S(text))

        return train

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


    def clear_cache(self):
        """
        手动清理缓存
        """
        if os.path.exists(self._model_path) is True:
            os.remove(self._model_path)
        if os.path.exists(self._train_vec_path) is True:
            os.remove(self._train_vec_path)


    def get_train_vec(self):
        """
        返回训练用到的向量，供分类器使用
        """
        if self._train_vec is None:
            self._load_vec()
        return self._train_vec
