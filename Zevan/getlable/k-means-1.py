import jieba, os, re
from jieba import analyse
from zhon.hanzi import punctuation
from zhon.pinyin import non_stops,stops
import time
import datetime
import codecs
from sklearn import feature_extraction
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import cluster
from sklearn.cluster import KMeans


#计算运行的时间
start = time.clock()
#停用词的文档
stop_words_file = 'E:\python\spider_aixue\\aixue_gensim\\all_stopword.txt'
#存储文件的文件夹
folder_path = 'E:\python\spider_aixue\\news\qqnews_2\\2017-08-27\首页'


#创建一个用来存储分词的全局变量
file_fenciresult = 'FenciResult.txt'
f1 = codecs.open(file_fenciresult,'w',encoding='utf-16')

#需要多少个中心点
number = 30
#类的标签为总数减1


#取出停用词
def get_stopWords(stopWords_fn):
    with open(stopWords_fn, 'rb') as f:
        stopWords_set = {line.strip().decode('utf-8') for line in f}
    return stopWords_set

#得到文件列表，顺便把每个文件的名字存储在一个文件夹里面
def get_file_list(folder_pth):
    file_name = "filenamelist.txt"
    Filelist = os.listdir(folder_pth)
    with open(file_name, 'w', encoding='utf-16') as f:
        for file in Filelist:
            f.write(str(file))
            f.write('\r\n')
    return file_name

#每个文档的处理和分词
def get_file_result(file_list):
    global folder_path
    global stop_words_file
    for file in file_list:
        if len(file) <3:
            continue
        jieba.load_userdict('dict.txt')
        #去除文档中的英文字符
        name = folder_path + '\\' + str(file)
        stopWords = get_stopWords((stop_words_file))
        with open(name, 'r', encoding='utf-16') as f:
            li = []
            for l in f.readlines():
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
            f1.write('\r\n')
            f1.write(' '.join(li))
            f1.write('\r\n\r\n')

def get_tfidfresult(Result_file):
    # 开始计算权重等
    # 写出词袋文档和数字文档
    # 由于有时文章过长文本可能自动分词
    corpus = []
    # 立一个flag
    write_flag = False

    with open(Result_file, 'r', encoding='utf-16') as f:
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

    vectorizer = CountVectorizer()
    transformer = TfidfTransformer()
    tfidf = transformer.fit_transform(vectorizer.fit_transform(corpus))
    word = vectorizer.get_feature_names()
    weight = tfidf.toarray()

    file_givereader = "Data_print.txt"
    with open(file_givereader, 'w', encoding='utf-16') as f:
        for i in range(len(weight)):
            f.write(u'--------这里输出第' + str(i+1) + u"---------类文本的词语tf-idf权重--------")
            f.write('\r\n')
            for j in range(len(word)):
                f.write(str(word[j]) + ':' + str(weight[i][j]))
                f.write('\r\n')
            f.write('\r\n')
    #方便之后的计算
    return weight


def get_label(weight):
    global number
    print("Begin KMeans")
    #number个中心点
    clf = KMeans(n_clusters=number)
    s = clf.fit_predict(weight)
    file_name = "Kmeans_data.txt"

    with open(file_name,'w',encoding='utf-16') as f:
        f.write('-----------每个文档的标签如下：--------------')
        f.write('\r\n')
        f.write(' '.join(list(str(s))))
        f.write('\r\n')
        for i in range(len(clf.cluster_centers_)):
            f.write(' '.join(str(clf.cluster_centers_[i])))
            f.write('\r\n')
        f.write('\r\n\r\n')

    print(clf.inertia_)
    #用于检测分类是否成功
    return s

def merge_namelist(filelist,lable):
    #首先判定是否成功
    if len(filelist) != len(lable):
        print('分类不成功！')
        exit(1)
    result_merge = []

    #创建一个文档，便于检测
    filename = "ResultMerge.txt"
    f = codecs.open(filename,'w',encoding='utf-16')

    for i in range(len(filelist)):
        st = str(filelist[i]) + '|' + str(lable[i])
        f.write(st)
        f.write('\r\n')

    return filename

def get_finalresult(file_result_merge):
    output = []
    file_name = "FinalResult.txt"
    file = codecs.open(file_name,'w',encoding='utf-16')
    with open(file_result_merge, 'r', encoding='utf-16') as f:
        for l in f.readlines():
            if len(l.strip()) < 3:
                continue
            value = l.split('|')
            output.append(value)

    for i in range(number):
        title ='-'*10  + "标签" + str(i+1) + '-'*10
        st = title + ':' + '\r\n'
        for j in range(len(output)):
            if int(output[j][1]) == i:
                st += str(output[j][0]) + '\t'
        file.write(st)
        file.write('\r\n')
        print(st)


def main():
    global folder_path
    file_listname = get_file_list(folder_path)
    file_list = []
    with open(file_listname,'r',encoding='utf-16') as f:
        for l in f.readlines():
            if len(l) < 3:
                continue
            file_list.append(l.strip())
    #开始分词
    get_file_result(file_list)
    f1.close()
    weight = get_tfidfresult(file_fenciresult)
    s = get_label(weight)
    if len(s) == len(file_list):
        print('\n分类成功。')
    else:
        print(len(s))
        print(len(file_list))
        print("\n分词文档存在错误")
        exit(1)
    resultmerge = merge_namelist(file_list,s)
    get_finalresult(resultmerge)
main()
end = time.clock()
t = end -start
print('耗费时间' + '\t' + str(t))
