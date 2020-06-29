#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

"""
函数说明:高斯朴素贝叶斯分类器
Functions:
	__init__ - 初始化
	values - 打印类成员信息
	obtainPrior - 计算先验概率矩阵
	obtainAverages - 计算均值矩阵
    obtainVariances - 计算方差矩阵
	obtainProbDensity - 计算条件概率矩阵
    setAll - 参数填充
    predictProb - 计算预测概率矩阵
    fitDenoise - 动态降噪训练
    fitBagging - 集成
(公用)
    fit - 训练高斯朴素贝叶斯分类器
    predict - 计算预测矩阵
"""
class GaussianNaiveBayesClassfier:
    """
    方法说明:初始化
    Parameters:
        has_denoise - 使用/不使用动态降噪 default:False
        bagging_rate - 附加分类器数量 default:0
        bagging_weakness - 附加分类器学习强度（1-训练集比例） default:0.3
    """
    def __init__(self, has_denoise=False, bagging_rate=0, bagging_weakness=0.3):
        self.prior = None                           # 先验概率矩阵
        self.averages = None                        # 各类均值矩阵
        self.variances = None                       # 各类方差矩阵
        self.n_class = None                         # 类别数量
        self.denoise = has_denoise                  # 动态降噪开关
        self.ignore_dim = None                      # 降噪后忽略的维数列表
        self.mix = bagging_rate                     # 附加分类器数量
        self.bagging_classifiers = None             # 附加分类器
        self.bagging_weakness = bagging_weakness    # 附加分类器强度

    """
    方法说明:打印类成员信息
    """
    def values(self):
        print('----In Classifier----')
        print('prior:')
        print(self.prior)
        print('averages:')
        print(self.averages)
        print('variances:')
        print(self.variances)
        print('n_class:')
        print(self.n_class)

    """
    方法说明:计算先验概率矩阵
    Parameters:
        dataFrame - 解析的数据集
    Returns:
        prior - 先验概率矩阵
            format:
                    [P0 P1 P2 P3 ... P(n-1)]
    """
    def obtainPrior(self, dataFrame):
        # 获取计数矩阵
        #print(dataFrame.iloc[:, -1].value_counts())
        labelCounts = np.array(dataFrame.iloc[:, -1].value_counts()).astype(np.float)
        prior = labelCounts / dataFrame.iloc[:, -1].size
        return prior

    """
    方法说明:计算均值矩阵
    Parameters:
        dataFrame - 解析的数据集
    Returns:
        均值矩阵
            format:
                    [[<----Avg(label==0)---->]
                    [<----Avg(label==1)---->]
                    ...
                    [<----Avg(label==n-1)---->]]
    """
    def obtainAverages(self, dataFrame):
        return np.array([dataFrame[dataFrame.iloc[:,-1]==i].mean(axis=0).iloc[0:-1] for i in range(self.n_class)])

    """
    方法说明:计算方差矩阵
    Parameters:
        dataFrame - 解析的数据集
    Returns:
        方差矩阵
            format:
                    [[<----Var(label==0)---->]
                    [<----Var(label==1)---->]
                    ...
                    [<----Var(label==n-1)---->]]
    """
    def obtainVariances(self,dataFrame):
        return np.array([dataFrame[dataFrame.iloc[:, -1] == i].var(axis=0).iloc[0:-1] for i in range(self.n_class)])

    """
    方法说明:计算条件概率矩阵
    Parameters:
        vector - 测试向量
    Returns:
        条件概率矩阵（非严格定义的条件概率，采用不带系数的概率密度矩阵代替）
            format:
                    [P(vector|label==0) P(vector|label==1) ... P(vector|label==n-1)]
    """
    def obtainProbDensity(self, vector):
        #print(vector)
        # 高斯分布下的各类各特征概率密度(不带1/sqrt(2*pi)系数)矩阵
        density = np.exp(-(vector - self.averages) ** 2 / (2 * self.variances)) / np.sqrt(self.variances)
        #print(density)
        # 贝叶斯独立假设下的各类条件概率密度矩阵(每行特征概率密度求积)
        unidensity = density.prod(axis=1)
        # 处理被约为0的数
        unidensity[unidensity==0.0] = 2.0 ** (-1023)
        return unidensity
    """
    方法说明:参数填充
    Parameters:
        dataFrame - 解析的数据集
    """
    def setAll(self, dataFrame):
        self.prior = self.obtainPrior(dataFrame)
        self.n_class = len(self.prior)
        self.averages = self.obtainAverages(dataFrame)
        self.variances = self.obtainVariances(dataFrame)

    """
    方法说明:计算预测概率矩阵
    Parameters:
        testFrame - 测试数据集
    Returns:
        jointProb - 预测概率矩阵（非严格定义的概率，不计算相同的分母）
            format:
                    [[P(label==0|vector0) P(label==1|vector0) ... P(label==n-1|vector0)]
                    [P(label==0|vector1) P(label==1|vector1) ... P(label==n-1|vector1)]
                    ...
                    [P(label==0|vector(n-1)) P(label==1|vector(n-1)) ... P(label==n-1|vector(n-1))]]
    """
    def predictProb(self, testFrame):
        #print(testFrame)
        if self.denoise == True:    # 去除动态降噪忽略的维数
            testFrame = testFrame.drop(self.ignore_dim, axis=1)
            #print(testFrame.shape)
        # testFrame测试数据的条件概率密度矩阵
        probDensity = np.apply_along_axis(self.obtainProbDensity, axis=1, arr=testFrame.values)
        #print(probDensity)
        # (标签,特征)联合概率密度矩阵
        jointProb = self.prior * probDensity
        # 总和归一化的(标签,特征)联合概率密度矩阵
        jointProb = jointProb / jointProb.sum(axis=1)[:,None]
        #print(jointProb)
        return jointProb

    """
    方法说明:动态降噪训练
    Parameters:
        trainFrame - 训练集
        regulateFrame - 验证集
    """
    def fitDenoise(self, trainFrame, regulateFrame):
        #pd.set_option('mode.chained_assignment', 'raise')
        self.ignore_dim = []
        delCol = -2 # 从除去最后的分类标签列的倒数第一列开始删除特征
        for i in range(trainFrame.shape[1] - 1):
            # 计算未删除当前特征时的验证集正确率
            self.setAll(trainFrame)
            #print(trainFrame.shape)
            subPredict = self.predictProb(regulateFrame.iloc[:, 0:-1]).argmax(axis=1)
            subRef = regulateFrame.iloc[:, -1].values
            diff = subPredict - subRef
            totGood = sum(diff == 0)
            decTrainFrame = trainFrame.drop(trainFrame.columns[delCol], axis=1)
            # 计算删除当前特征后的验证集正确率
            self.ignore_dim.append(trainFrame.columns[delCol])
            self.setAll(decTrainFrame)
            subPredict = self.predictProb(regulateFrame.iloc[:, 0:-1]).argmax(axis=1)
            subRef = regulateFrame.iloc[:, -1].values
            diff = subPredict - subRef
            totGoodDel = sum(diff == 0)
            if totGood < totGoodDel:    # 删除后正确率更高时
                #print(trainFrame.columns[delCol])
                trainFrame = trainFrame.drop(trainFrame.columns[delCol], axis=1)    # 去除该特征
            else:   # 删除前正确率更高时
                delCol = delCol - 1 # 考察前一个特征
                self.ignore_dim.pop(-1) # 回溯到前一状态
            #print('未删节正确数：%d, 删节后正确数：%d' % (totGood, totGoodDel))
        self.setAll(trainFrame) # 去除特征完毕后重新训练
        #print(self.ignore_dim)

    """
    方法说明:集成
    Parameters:
        trainFrame - 训练集
    """
    def fitBagging(self, trainFrame):
        self.bagging_classifiers = []
        for i in range(self.mix):   # 创建附加分类器
            self.bagging_classifiers.append(GaussianNaiveBayesClassfier(has_denoise=self.denoise, bagging_rate=0))
        for classifier in self.bagging_classifiers: # 根据分类器强度划分训练/验证集
            splitedTrainFrame, nouseFrame = train_test_split(trainFrame.copy(), test_size=self.bagging_weakness)
            classifier.fit(splitedTrainFrame)

    """
    方法说明:训练高斯朴素贝叶斯分类器
    Parameters:
        trainFrame - 训练集
    """
    def fit(self, trainFrame):
        if self.denoise == True:
            # 为动态降噪划分训练/验证集
            splitedTrainFrame, regulateFrame = train_test_split(trainFrame.copy(), test_size=0.3)
            #print(splitedTrainFrame.shape)
            # 动态降噪训练
            self.fitDenoise(splitedTrainFrame, regulateFrame)
            #print(splitedTrainFrame.shape)
        else:
            self.setAll(trainFrame.copy())
        if self.mix > 0:
            # 附加分类器训练
            self.fitBagging(trainFrame.copy())

    """
    方法说明:计算预测矩阵
    Parameters:
        testFrame - 测试集
        format - 返回格式
            format: 'label' - 返回预测标签矩阵
                    'prob' - 返回预测概率矩阵
    """
    def predict(self, testFrame, format='label'):
        probArr = self.predictProb(testFrame)
        if self.mix > 0:    # 累加各分类器返回的预测概率矩阵
            for classifier in self.bagging_classifiers:
                probArr = probArr + classifier.predictProb(testFrame)
                #print(classifier.predictProb(testFrame))
        #print(probArr)
        if format == 'label':
            return probArr.argmax(axis=1)   # 返回预测标签矩阵
        elif format == 'prob':
            return probArr / probArr.sum(axis=1)[:,None] # 返回预测概率矩阵
        else:
            print('format not matched, only \'label\' or \'prob\' is valid')
            exit(-1)
