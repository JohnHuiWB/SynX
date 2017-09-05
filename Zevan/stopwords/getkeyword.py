'''
请把停用词文件下载好
请直接调用getkey_main，并输入放好文件的文件夹
'''



import os
import jieba
import codecs
from itertools import islice
import re
from zhon.hanzi import punctuation
from zhon.pinyin import non_stops,stops
from sklearn import feature_extraction
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np


class Getkeyword(object):
    def __init__(self):
        self.path = os.getcwd()
        self.top = 6
        self.num = 1
        self.stopWords = 'all_stopword.txt'
        self.folder = ''

        #自动处理的量
        self.fenciresult = []
        self.filenamelist = []

    # 创建文件夹
    def mkfolder(self):
        global folderName
        folderName = self.path + '\\' + 'experiment' + str(self.num)
        try:
            if os.path.exists(folderName) == True:
                pass
            else:
                os.mkdir(folderName)
            print("文件夹创建成功")
        except Exception as e:
            print(e)

    # 获取停用词
    def get_stopWords(self,stopWords_fn=None):
        if stopWords_fn == None:
            if os.path.exists(self.stopWords) == False:
                print('请导入停用词文件。')
            stopWords_fn = self.stopWords
        with open(stopWords_fn, 'rb') as f:
            stopWords_set = {line.strip().decode('utf-8') for line in f}
        return stopWords_set


    def get_fileNamelist(self,target_folder=None):
        if target_folder == None:
            print('请输入你要分析的文件夹！')
            exit(1)

        self.folder = target_folder
        filename_list = []
        for name in os.listdir(target_folder):
            filename_list.append(name)

        self.filenamelist = filename_list
        return filename_list

    def get_fenciResult(self):
        # 第一步读取文档，你是列表这里可以略过
        namelist = self.filenamelist
        if os.path.exists('dict.txt') == False:
            pass
        else:
            jieba.load_userdict('dict.txt')

        # 创建停用词表
        stopWords = self.get_stopWords()
        # 好了列表做好
        fenci = []
        for name in namelist:
            st = self.folder + '\\' + str(name)
            with open(st, 'r', encoding='utf-16') as f3:
                li = []
                for l in islice(f3, 1, None):
                    # 处理文本中无关的符号
                    l = re.sub(r'[A-Za-z\\]+', '', l)
                    r = r"[\s+\.\!\/_,$%^*(+\"\']+|[+——！，。？、~@#￥%……&*（）]+"
                    l = re.sub(r"[%s+]" % punctuation, "", l)
                    l = re.sub(r"[%s+]" % non_stops, '', l)
                    l = re.sub(r"[%s+]" % stops, '', l)
                    l = re.sub(r, '', l)
                    l = l.strip()
                    l = re.sub(r"\d{5,6000}", '', l)
                    if l == '':
                        continue
                    seg_list = jieba.cut(l, cut_all=False)

                    # 利用结巴分词去除停用词
                    for word in seg_list:
                        if word not in stopWords:
                            if word != '\t':
                                li.append(word)
                fenci.append(str(' '.join(li)))
        self.fenciresult = fenci
        return fenci


    def get_WeightandWords(self):
        # 这里还是文件的操作
        corpus = self.fenciresult

        # 这里的corpus就是文本分词之后的结果

        vectorizer = CountVectorizer()
        transformer = TfidfTransformer()
        # 计算
        tfidf = transformer.fit_transform(vectorizer.fit_transform(corpus))
        # 结果
        weight = tfidf.toarray()
        words = vectorizer.get_feature_names()
        return weight, words

    def print_Result(self):
        top = self.top
        namelist = self.filenamelist
        weight, words = self.get_WeightandWords()

        # 稍微处理一下文件的名字格式，让文件更加美观
        titlelist = []
        for name in self.filenamelist:
            name = str(name).replace('.txt', '')
            titlelist.append(name)

        if len(titlelist) == len(weight):
            print("成功")
        else:
            print(len(titlelist))
            print(len(weight))
            print("不成功")

        mi = len(titlelist)
        if len(titlelist) > len(weight):
            mi = len(weight)

        #存储最后的结果,final有top+1行name,keyword,
        final = []
        for j in range(mi):
            li = []
            print(u'{}'.format(titlelist[j]))
            li.append(titlelist[j])
            # 排序
            loc = np.argsort(-weight[j])
            for i in range(top):
                print('-{a}：{b} {c}'.format(a=str(i + 1), b=words[loc[i]], c=weight[j][loc[i]]))
                li.append(words[loc[i]])
            final.append(li)
            print('\n')
        return  final

    def getkey_main(self, stopWords_fn=None,target_folder = None):
        self.get_stopWords(stopWords_fn)
        self.get_fileNamelist(target_folder)
        self.get_fenciResult()
        self.print_Result()
