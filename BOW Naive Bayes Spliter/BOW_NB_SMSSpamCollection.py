# -*- coding: UTF-8 -*-
import numpy as np
import re
import random

"""
函数说明:将切分的实验样本词条整理成不重复的词条列表，也就是词汇表
Parameters:
    dataSet - 整理的样本数据集
Returns:
    vocabSet - 返回不重复的词条列表，也就是词汇表
"""
def createVocabList(dataSet, hasStop = 0, stopList = []):
    vocabSet = set([])  # 创建一个空的不重复列表
    for document in dataSet:
        vocabSet = vocabSet | set(document)  # 取并集
    if (hasStop): vocabSet = vocabSet - set(stopList)
    print(vocabSet)
    return list(vocabSet)

"""
函数说明:根据vocabList词汇表，构建词袋模型
Parameters:
    vocabList - createVocabList返回的列表
    inputSet - 切分的词条列表
Returns:
    returnVec - 文档向量,词袋模型
"""
def bagOfWords2Vec(vocabList, inputSet):
    returnVec = [0] * len(vocabList)  # 创建一个其中所含元素都为0的向量
    for word in inputSet:             # 遍历每个词条
        if word in vocabList:         # 如果词条存在于词汇表中，则计数加一
            returnVec[vocabList.index(word)] += 1
        else:
            #print("the word: %s is not in my Vocabulary!" % word)
            pass
    return returnVec  # 返回词袋模型

"""
函数说明:朴素贝叶斯分类器训练函数
Parameters:
    trainMatrix - 训练文档矩阵，即setOfWords2Vec返回的returnVec构成的矩阵
    trainCategory - 训练类别标签向量，即loadDataSet返回的classVec
Returns:
    p0Vect - 正常邮件类的条件概率数组
    p1Vect - 垃圾邮件类的条件概率数组
    pAbusive - 文档属于垃圾邮件类的概率
"""
def trainNB0(trainMatrix, trainCategory):
    numTrainDocs = len(trainMatrix)  # 计算训练的文档数目
    numWords = len(trainMatrix[0])  # 计算每篇文档的词条数
    pAbusive = sum(trainCategory) / float(numTrainDocs)  # 文档属于垃圾邮件类的概率
    p0Num = np.ones(numWords)
    p1Num = np.ones(numWords)  # 创建numpy.ones数组,词条出现数初始化为1,拉普拉斯平滑
    p0Denom = 2.0
    p1Denom = 2.0  # 分母初始化为2 ,拉普拉斯平滑
    for i in range(numTrainDocs):
        if trainCategory[i] == 1:  # 统计属于垃圾邮件类的条件概率所需的数据，即P(w0|1),P(w1|1),P(w2|1)···
            p1Num += trainMatrix[i]
            p1Denom += sum(trainMatrix[i])
        else:  # 统计属于正常邮件类的条件概率所需的数据，即P(w0|0),P(w1|0),P(w2|0)···
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])
    p1Vect = np.log(p1Num / p1Denom)
    p0Vect = np.log(p0Num / p0Denom)   #取对数，防止下溢出
    return p0Vect, p1Vect, pAbusive  # 返回属于正常邮件类的条件概率数组，属于垃圾邮件类的条件概率数组，文档属于垃圾邮件类的概率

"""
函数说明:朴素贝叶斯分类器分类函数
Parameters:
	vec2Classify - 待分类的词条数组
	p0Vec - 正常邮件类的条件概率数组
	p1Vec - 垃圾邮件类的条件概率数组
	pClass1 - 文档属于垃圾邮件的概率
Returns:
	0 - 属于正常邮件类
	1 - 属于垃圾邮件类
"""
def classifyNB(vec2Classify, p0Vec, p1Vec, pClass1):
    p1=sum(vec2Classify*p1Vec)+np.log(pClass1)
    p0=sum(vec2Classify*p0Vec)+np.log(1.0-pClass1)
    if p1 > p0:
        return 1
    else:
        return 0

"""
函数说明:接收一个大字符串并将其解析为字符串列表
"""
def textParse(bigString):  # 将字符串转换为字符列表
    listOfTokens = re.split(r'\W+', bigString)  # 将特殊符号作为切分标志进行字符串切分，即非字母、非数字
    return [tok.lower() for tok in listOfTokens]  # 单词变成小写

"""
函数说明:解析SMSSpamCollection数据集
Parameters:
	filePath - 数据集路径
	validLen - 保留单词最小长度
Returns:
	docList - 解析的样本数据集
	classList - 解析的样本标签集
"""
def decodeCollection(filePath, validLen):
    docList = []
    classList = []
    lines = open(filePath, 'r', encoding='utf-8').readlines() # UTF-8编码格式
    for line in lines:  # 读取每一行
        tempList = textParse(line)
        if (tempList[0] == 'ham'): classList.append(1)
        else: classList.append(0)
        tempList.pop(0)
        tempList = [tok for tok in tempList if len(tok) > validLen]
        docList.append(tempList)
    return docList, classList

"""
函数说明:测试朴素贝叶斯分类器，使用朴素贝叶斯进行交叉验证，90%作为训练集,10%作为测试集
Parameters:
	sampleOffset - 数据集划分方法（标定测试集第一个元素在数据集中的偏移），取负值时随机划分
	validLen - 保留单词最小长度
	hasStop - 是否有停用词表
	stopList - 停用词表
"""
def spamTest(sampleOffset = 0,validLen = 0, hasStop = 0, stopList = []):
    docList, classList = decodeCollection('email/SMSSpamCollection', validLen)
    vocabList = createVocabList(docList, hasStop, stopList)  # 创建词汇表，不重复
    mailNum = len(docList)
    trainingSet = list(range(mailNum))
    testSet = []  # 创建存储训练集的索引值的列表和测试集的索引值的列表
    for i in range(int(mailNum / 10)):  # 从mailNum个邮件中，随机挑选出90%作为训练集,10%作为测试集
        if sampleOffset < 0: sampleIndex = int(random.uniform(0, len(trainingSet)))  # 随机选取索引值，也可以试试randIndex = i + 2000
        else: sampleIndex = i + sampleOffset
        testSet.append(trainingSet[sampleIndex])  # 添加测试集的索引值
        del (trainingSet[sampleIndex])  # 在训练集列表中删除添加到测试集的索引值
    trainMat = []
    trainClasses = []  # 创建训练集矩阵和训练集类别标签系向量
    for docIndex in trainingSet:  # 遍历训练集
        trainMat.append(bagOfWords2Vec(vocabList, docList[docIndex]))  # 将生成的词袋模型添加到训练矩阵中
        trainClasses.append(classList[docIndex])  # 将类别添加到训练集类别标签系向量中
    p0V, p1V, pSpam = trainNB0(np.array(trainMat), np.array(trainClasses))  # 训练朴素贝叶斯模型
    errorCount = 0  # 错误分类计数
    for docIndex in testSet:  # 遍历测试集
        wordVector = bagOfWords2Vec(vocabList, docList[docIndex])  # 测试集的词袋模型
        if classifyNB(np.array(wordVector), p0V, p1V, pSpam) != classList[docIndex]:  # 如果分类错误
            errorCount += 1  # 错误计数加1
            print("分类错误的测试集", docIndex, "：", docList[docIndex])
    print('错误率：%.2f%%' % (float(errorCount) / len(testSet) * 100))


if __name__ == '__main__':
    # 停用词表
    stopList = ['very', 'ourselves', 'am', 'doesn', 'through', 'me', 'against', 'up', 'just', 'her', 'ours',
                'couldn', 'because', 'is', 'isn', 'it', 'only', 'in', 'such', 'too', 'mustn', 'under', 'their',
                'if', 'to', 'my', 'himself', 'after', 'why', 'while', 'can', 'each', 'itself', 'his', 'all', 'once',
                'herself', 'more', 'our', 'they', 'hasn', 'on', 'ma', 'them', 'its', 'where', 'did', 'll', 'you',
                'didn', 'nor', 'as', 'now', 'before', 'those', 'yours', 'from', 'who', 'was', 'm', 'been', 'will',
                'into', 'same', 'how', 'some', 'of', 'out', 'with', 's', 'being', 't', 'mightn', 'she', 'again', 'be',
                'by', 'shan', 'have', 'yourselves', 'needn', 'and', 'are', 'o', 'these', 'further', 'most', 'yourself',
                'having', 'aren', 'here', 'he', 'were', 'but', 'this', 'myself', 'own', 'we', 'so', 'i', 'does', 'both',
                'when', 'between', 'd', 'had', 'the', 'y', 'has', 'down', 'off', 'than', 'haven', 'whom', 'wouldn',
                'should', 've', 'over', 'themselves', 'few', 'then', 'hadn', 'what', 'until', 'won', 'no', 'about',
                'any', 'that', 'for', 'shouldn', 'don', 'do', 'there', 'doing', 'an', 'or', 'ain', 'hers', 'wasn',
                'weren', 'above', 'a', 'at', 'your', 'theirs', 'below', 'other', 'not', 're', 'him', 'during', 'which']

    spamTest(999, 0, 0) # 3.23
    spamTest(999, 1, 0) # 2.87
    spamTest(999, 2, 0) # 4.85
    spamTest(999, 0, 1, [tok for tok in stopList if len(tok) < 2]) # 2.51
    spamTest(999, 0, 1, [tok for tok in stopList if len(tok) < 3]) # 3.23
    spamTest(999, 1, 1, stopList) # 3.77