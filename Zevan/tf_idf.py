# conding : utf-8
"""
import jieba.posseg as pseg
import re

def tf_fileopen(file_name):
    file = open(file_name,'r')
    f = file.read()
    f = f.strip()
    return f

file_name = "D:\\learngit\\github学习笔记.txt"
words = tf_fileopen(file_name)
print(words)
words = pseg.cut(words)
for key in words :
    print (key.word,key.flag)
"""

import jieba
import jieba.posseg as pseg
import os
import sys
from sklearn import feature_extraction
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
import numpy
import scipy.sparse
import theano

def tf_fileopen(file_name):
    file = open(file_name,'r')
    f=[]
    for line in file.readlines():
        for word in jieba.cut(line.strip()):
            f.append(word)
    return f

jieba.load_userdict("dict.txt")
file_name = "D:\learngit\\github学习笔记.txt"
corpus = tf_fileopen(file_name)

if __name__ == '__main__':
    vector = CountVectorizer()
    transform = TfidfTransformer()
    tfidf = transform.fit_transform(vector.fit_transform(corpus))
    word = vector.get_feature_names()
    weight = tfidf.toarray()
    for i in range(len(weight)):
        print(u"----这里输出第", i, u"类词语的权重----")
        for j in range(len(word)):
            print(word[j], weight[i][j])
