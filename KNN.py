import jieba
from jieba import analyse
import os
import re
from numpy import *
import operator


postingList =[]
tfidf = analyse.extract_tags
name = os.listdir('F:\python\作业\测试txt')

def getDataSet(): #处理文本
    #i = random.randint(0,19)
    f = open('F:\python\作业\新建测试集\\' + name[0],'r') #读入txt文件
    ftxt = f.read()
    txt = re.sub("[\s+\.\!\/_,$%^*(+\"\']+|[+——！，。？、~@#￥%……&*（）]+", "",ftxt) #去除停用词
    txt ="/".join(jieba.cut(txt))
    features= tfidf(txt,50) #提取特征
    List=[]
    for feature in features:
        List.append(feature) 
    return List

def getDataSet2(): #处理文本
    #i = random.randint(0,19)
    f = open('F:\python\作业\新建测试集\\' + name[1],'r') #读入txt文件
    ftxt = f.read()
    txt = re.sub("[\s+\.\!\/_,$%^*(+\"\']+|[+——！，。？、~@#￥%……&*（）]+", "",ftxt) #去除停用词
    txt ="/".join(jieba.cut(txt))
    features= tfidf(txt,50) #提取特征
    List=[]
    for feature in features:
        List.append(feature) 
    return List

def loadDataSet(): #加载文本
    i =0
    while(i<20):
        f = open('F:\python\作业\测试txt\\' + name[i],'r') #读入txt文件
        ftxt = f.read()
        txt = re.sub("[\s+\.\!\/_,$%^*(+\"\']+|[+——！，。？、~@#￥%……&*（）]+", "",ftxt) #去除停用词
        txt ="/".join(jieba.cut(txt))
        features= tfidf(txt,50) #提取特征
        List=[]
        for feature in features:
            List.append(feature)
        postingList.append(List)
        List=List[:0]
        i =i+1
    classVec=[0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1]  #0是正常文本，1不是
    return postingList,classVec

def createVocabList(dataSet):  #创建一个带所有单词的列表
    vocabSet =set([])
    for document in dataSet:
        vocabSet = vocabSet | set(document) 
    return list(vocabSet)

def bagofWords2VecMN(vocabList,inputSet):  #词袋模型
    returnVec=[0]*len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)]+=1
    return returnVec

def classify(testdata,dataSet,labels,k):
    '''
    输入：testdata：未知数据的向量
          dataSet:m个已知数据的向量
          labels：M个已知数据标签的向量
          k：选择前K个最相似的数据(k一般是不大于20的整数)
    输出：未知数据的标签
    '''
    dataSetSize = dataSet.shape[0] #获得已知数据M的大小
    a=tile(testdata,(dataSetSize,1))
    #用tile函数扩展testdata为N*M的矩阵，并于已知数据做减法
    diffMat = a-dataSet
    sqDiffMat = diffMat**2 #平方
    sqDistances = sqDiffMat.sum(axis=1) #axis=0,表示列，axis =1,表示行。
    distances = sqDistances**0.5 #开方
    indexes =distances.argsort() #返回从小到大的值的索引
    labelCount={} #创建计数字典
    for i in range(k):  #选择距离最小的k个标签
        label =labels[indexes[i]]
        labelCount[label] =labelCount.get(label,0)+1
    #sorted函数返回一个由tuple组成的list，如[(0,3)]    
    sortedLabelCount =sorted(labelCount.items(),key = operator.itemgetter(1),reverse =True)
    x =sortedLabelCount[0][0]
    return x


# file2matrix() 返回一个特征矩阵和一个标签矩阵效果与上面array（classVec)和array（trainMat）一样
def testingKNN():
    listOPosts,listClasses = loadDataSet()
    myVocabList = createVocabList(listOPosts)
    trainMat=[]
    for postinDoc in listOPosts:
        trainMat.append(bagofWords2VecMN(myVocabList, postinDoc))
    testEntry = getDataSet()
    thisDoc = array(bagofWords2VecMN(myVocabList, testEntry))
    classifierResult =classify(thisDoc,array(trainMat),array(listClasses),3)
    print(testEntry,'classified as:',classifierResult)
    testEntry = getDataSet2()
    thisDoc = array(bagofWords2VecMN(myVocabList, testEntry))
    classifierResult =classify(thisDoc,array(trainMat),array(listClasses),3)
    print(testEntry,'classified as:',classifierResult)


testingKNN()
    
    
