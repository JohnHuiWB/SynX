import jieba
from jieba import analyse
import os
import re
from numpy import *

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
        print(List)
        List=List[:0]
        i =i+1
    classVec=[0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1]  #0是正常文本，1不是
    return postingList,classVec

def createVocabList(dataSet):  #创建一个带所有单词的列表
    vocabSet =set([])
    for document in dataSet:
        vocabSet = vocabSet | set(document) 
    return list(vocabSet)

def setOfWords2Vec(vocabList,inputSet):
    returnVec = [0]*len(vocabList) #创建一个所包含元素都为0的向量
    for word in inputSet:  #遍历文档中的所有单词，如果出现了词汇表中的单词，则将输出的文档向量中的对应值设置为1
        if word in vocabList:
            returnVec[vocabList.index(word)]=1
        else:
            print("the word：%s is not in my Vocabulary"% word)
    return returnVec
'''
 我们将每个词的出现与否作为一个特征，这可以被描述为词集模型(set-of-words model)。
 如果一个词在文档中出现不止一次，这可能意味着包含该词是否出现在文档中所不能表达的某种信息,
 这种方法被称为词袋模型(bag-of-words model)。
 在词袋中，每个单词可以出现多次，而在词集中，每个词只能出现一次。
 为适应词袋模型，需要对函数setOfWords2Vec稍加修改，修改后的函数称为bagOfWords2VecMN
'''

def bagofWords2VecMN(vocabList,inputSet):  #
    returnVec=[0]*len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)]+=1
    return returnVec

'''
#测试
dataSet,classes =loadDataSet()
print(dataSet)
print("\n")
vocabList = createVocabList(dataSet)
print(vocabList)
print("\n")
setWordsVec=setOfWords2Vec(vocabList,dataSet[0])
print(setWordsVec)
print("\n")
'''

def trainNB0(trainMatrix,trainCategory):
    '''
    朴素贝叶斯分类器训练函数(此处仅处理两类分类问题)
    trainMatrix:文档矩阵
    trainCategory:每篇文档类别标签
    '''
    numTrainDocs = len(trainMatrix)
    numWords = len(trainMatrix[0])
    pAbusive = sum(trainCategory)/float(numTrainDocs)
    #初始化所有词出现数为1，并将分母初始化为2，避免某一个概率值为0
    p0Num = ones(numWords); p1Num = ones(numWords)#
    p0Denom = 2.0; p1Denom = 2.0 #
    for i in range(numTrainDocs):
        if trainCategory[i] == 1:
            p1Num += trainMatrix[i]
            p1Denom += sum(trainMatrix[i])
        else:
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])
    #将结果取自然对数，避免下溢出，即太多很小的数相乘造成的影响
    p1Vect = log(p1Num/p1Denom)#change to log()
    p0Vect = log(p0Num/p0Denom)#change to log()
     # 返回值分别代表
    # 类别0中词的条件概率向量
    # 类别1中词的条件概率向量
    # 0和1类别发生概率向量（二维）
    return p0Vect,p1Vect,pAbusive
'''
#测试
dataSet,classes = loadDataSet()
vocabList = createVocabList(dataSet)
trainMat = []
for item in dataSet:
    trainMat.append(setOfWords2Vec(vocabList,item))
                    
p0v,p1v,pAb = trainNB0(trainMat,classes)
print(p0v)
print(p1v)
print(pAb)
'''
def classifyNB(vec2Classify, p0Vec, p1Vec, pClass1):
    '''
    分类函数
    vec2Classify:要分类的向量
    p0Vec, p1Vec, pClass1:分别对应trainNB0计算得到的3个概率
    '''
    p1 = sum(vec2Classify * p1Vec) + log(pClass1)
    p0 = sum(vec2Classify * p0Vec) + log(1.0 - pClass1)
    if p1 > p0:
        return 1
    else: 
        return 0
    
def testingNB():
    listOPosts,listClasses = loadDataSet()
    myVocabList = createVocabList(listOPosts)
    trainMat=[]
    for postinDoc in listOPosts:
        trainMat.append(bagofWords2VecMN(myVocabList, postinDoc))
    #训练模型，注意此处使用array.
    print("测试")
    print(array(trainMat))
    p0V,p1V,pAb = trainNB0(array(trainMat),array(listClasses))
    testEntry = getDataSet()
    thisDoc = array(bagofWords2VecMN(myVocabList, testEntry))
    print(testEntry,'classified as: ',classifyNB(thisDoc,p0V,p1V,pAb))
    testEntry = getDataSet2()
    thisDoc = array(bagofWords2VecMN(myVocabList, testEntry))
    print(testEntry,'classified as: ',classifyNB(thisDoc,p0V,p1V,pAb))


#测试
testingNB()



    
