#ok 我的习惯是创建一个本地文件夹来储存一些文档
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

#首先我们需要明确当前的目录
current_path = os.getcwd()
#全局变量实验的目录，但是这个执行文件不在文件夹的里面
folderName = ''
#第几次实验的次数
num = 1
#所用过的文本的名字
usedFile =[]
#文件夹目录
target_folder = 'E:\python\spider_aixue\\news\qqnews_2\\2017-09-01\首页'


#创建文件夹
def mkfolder():
    global folderName
    folderName = current_path + '\\' + 'experiment' + str(num)
    try:
        if os.path.exists(folderName) == True:
            pass
        else:
            os.mkdir(folderName)
        print("文件夹创建成功")
    except Exception as e:
        print(e)


#获取停用词，这个以前说过，我就不说了，自己准备好
def get_stopWords(stopWords_fn = None):
    if stopWords_fn == None:
        stopWords_fn = 'E:\python\spider_aixue\\aixue_gensim\\all_stopword.txt'
    with open(stopWords_fn, 'rb') as f:
        stopWords_set = {line.strip().decode('utf-8') for line in f}
    return stopWords_set


#我的习惯是进行文本的读写，所以这里会有大量文本的操作
def get_fileNamelist(target_folder=None):
    if target_folder == None:
        print('请输入你要分析的文件夹！')
        exit(1)
    file_name = folderName + '\\' + "File_namelist.txt"
    usedFile.append(file_name)
    #文件的操作
    file = codecs.open(file_name,'w',encoding='utf-16')
    for name in os.listdir(target_folder):
        file.write(str(name))
        file.write('\r\n')
    #注意这里的路径，和我一样玩文件的小伙伴请注意
    return file_name

def get_fenciResult(fileNamelist):
    '''
    我是文件操作主要是为了读取一个列表
    还有就是，我因为提前爬取了文件，文件名字是很标准的没有英文符的干扰
    如果你的文件名字有干扰，建议采用等一会的手法去除
    '''

    #第一步读取文档，你是列表这里可以略过
    namelist = []
    with open(fileNamelist,'r',encoding='utf-16') as f1:
        for l in f1.readlines():
            l = l.strip()
            namelist.append(l)

    #创建一个文件夹储存分词的文档
    file_name = folderName + '\\' + 'FenciResult.txt'
    f2 = codecs.open(file_name,'w',encoding='utf-16')
    #创建停用词表
    stopWords = get_stopWords()
    #好了列表做好
    for name in namelist:
        st = target_folder + '\\' + str(name)
        with open(st,'r',encoding='utf-16') as f3:
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
                li += list(seg_list)

                # 利用结巴分词去除停用词
                for word in seg_list:
                    if word not in stopWords:
                        if word != '\t':
                            li.append(word)

            #写入分词的结果的文档
            #注意文章过长有时需要分阶段录入
            f2.write('\r\n')
            f2.write(' '.join(li))
            f2.write('\r\n\r\n')
    return file_name


def get_WeightandWords(FenciResult):
    #这里还是文件的操作
    corpus = []
    # 立一个flag
    write_flag = False

    with open(FenciResult, 'r', encoding='utf-16') as f:
        front_s = ''
        s = ''
        for l in f.readlines():
            if len(l.strip()) < 3:
                write_flag = False
                s = ''
            else:
                write_flag = True
                s += str(l)
            if write_flag == False and len(front_s) > 3:
                corpus.append(front_s)
            front_s = s
    #这里的corpus就是文本分词之后的结果

    vectorizer = CountVectorizer()
    transformer = TfidfTransformer()
    #计算
    tfidf = transformer.fit_transform(vectorizer.fit_transform(corpus))
    #结果
    weight = tfidf.toarray()
    words = vectorizer.get_feature_names()
    return weight,words

def print_Result(weight, words,fileNamelist):
    top = 6
    namelist = []
    with open(fileNamelist,'r',encoding='utf-16') as f1:
        for l in f1.readlines():
            l = l.strip()
            namelist.append(l)

    #稍微处理一下文件的名字格式，让文件更加美观
    titlelist = []
    for name in namelist:
        name = str(name).replace('.txt','')
        titlelist.append(name)



    if len(titlelist) == len(weight):
        print("成功")
    else:
        print(len(titlelist))
        print(len(weight))
        print("不成功")

    mi = len(titlelist)
    print(titlelist)
    if len(titlelist) > len(weight):
        mi = len(weight)
        #排序
    for j in range(mi):
        print(u'{}'.format(titlelist[j]))
        loc = np.argsort(-weight[j])
        for i in range(top):
            print('-{a}：{b} {c}'.format(a=str(i+1),b = words[loc[i]], c = weight[j][loc[i]]))
        print('\n')



def main():
    mkfolder()

    FileNamelist = get_fileNamelist(target_folder)
    FenciResult = get_fenciResult(FileNamelist)
    weight, words = get_WeightandWords(FenciResult)
    print_Result(weight,words,FileNamelist)

main()
