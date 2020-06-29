#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

"""
函数说明:数据集生成，读取所有数据，划分成若干份训练/测试数据csv文件
Parameters:
	csvFile - 原始数据集csv文件名 default:'voice/voice.csv'
	num - 生成的训练/测试数据csv文件数量 default:1
"""
def createDataset(csvFile='voice/voice.csv', num=1):
    df = pd.read_csv(csvFile)   # 读取csv-->dataFrame
    trainFrames, testFrames = divideDataset(df, num)    # 划分num份训练/测试集
    for i in range(len(trainFrames)):   # 写入训练/测试集dataFrame-->csv
        trainFrames[i].to_csv('voice/voice_train_%d.csv' % (i + 1), index=False)
        testFrames[i].to_csv('voice/voice_test_%d.csv' % (i + 1), index=False)

"""
函数说明:读取数据集，读取一个csv文件
Parameters:
	csvFile - 数据集csv文件名 default:'voice/voice.csv'
Returns:
	dataFrame - 解析的数据集
"""
def loadDataset(csvFile='voice/voice.csv'):
    dataFrame = pd.read_csv(csvFile)    # 读取csv-->dataFrame
    #print(dataFrame)
    return dataFrame

"""
函数说明:处理缺省值和标签数值化
Parameters:
	dataFrame - 解析的数据集
	dicard - 舍弃带有缺省值的行或填充平均值 default:False
"""
def fixDeafults(dataFrame, discard=False):
    for index, row in dataFrame.iterrows():
        for i in range(len(row)-1):
            if row[i] == 0.0:   # 处理缺省值
                dataFrame.iloc[index, i] = np.nan   # 缺省值统一置为NaN
        # 标签二值化
        if row[-1] == 'male': dataFrame.iloc[index, -1] = 1 # male标签置1
        else: dataFrame.iloc[index, -1] = 0 # female标签置0
    #print(dataFrame)
    #print(dataFrame.mean())
    # 根据参数dicard处理
    if discard: dataFrame.dropna(axis=0, inplace=True)  # 舍弃带缺省值的行
    else:
        dataFrame.fillna(dataFrame.mean(), inplace=True)  # 缺省值填充均值
        dataFrame.dropna(axis=1, inplace=True)  # 舍弃带缺省值的列
    #print(dataFrame)
    #print(dataFrame.mean())

"""
函数说明:最大最小值标准化
Parameters:
	dataFrame - 解析的数据集
Returns:
	标准化的数据集
"""
def scaleMinMax(dataFrame):
    return (dataFrame - dataFrame.min()) / (dataFrame.max() - dataFrame.min())


"""
函数说明:正态分布标准化
Parameters:
	dataFrame - 解析的数据集
Returns:
	标准化的数据集
"""
def scaleNorm(dataFrame):
    return (dataFrame - dataFrame.mean()) / dataFrame.std()

"""
函数说明:划分数据集
Parameters:
	dataFrame - 解析的数据集
	num - 划分数量 default:1
	testSize - 测试集比例 default:0.3
Returns:
	trainFrames - 训练集列表
	testFrames - 测试集列表
"""
def divideDataset(dataFrame, num=1, testSize=0.3):
    trainFrames = []
    testFrames = []
    for i in range(num):
        train, test = train_test_split(dataFrame.copy(), test_size=testSize)
        trainFrames.append(train)
        testFrames.append(test)
    return trainFrames, testFrames  # 返回dataFrame列表

"""
函数说明:主特征提取
Parameters:
	trainFrame - 训练集
	testFrame - 测试集
	feature_num - 提取特征数量
	method - 提取特征方法 default:'var'
	    method: 'var' - 类间方差最大
	            'mean' - 类间相对均值差最大
	            'boost_weight' - 梯度提升决策树权重最大
	            'boost_gain' - 梯度提升决策树增益最大
	            others - 不提取特征
Returns:
	trainFrame[colName].copy() - 已提取特征的训练集切片
	testFrame[colName].copy() - 已提取特征的测试集切片
"""
def generateMainFeatures(trainFrame, testFrame, feature_num, method='var'):
    colName = trainFrame.columns.values.tolist()    # 获取列标签列表
    stdTrainFrame = scaleMinMax(trainFrame) # dataFrame最大最小值归一化
    if method == 'mean':    # 以类间相对均值差最大为标准提取主要特征
        # 获取male各项特征均值列表
        maleSampleMean = list(stdTrainFrame[stdTrainFrame.iloc[:,-1]==1].mean(axis=0).iloc[0:-1])
        # 获取female各项特征均值列表
        femaleSampleMean = list(stdTrainFrame[stdTrainFrame.iloc[:,-1]==0].mean(axis=0).iloc[0:-1])
        # 求类间相对均值差列表
        different = [abs(maleSampleMean[i] - femaleSampleMean[i]) / min(maleSampleMean[i], femaleSampleMean[i]) for i in range(len(maleSampleMean))]
        # 获取降序特征索引
        index = list(np.argsort(different)[-1:-feature_num-1:-1])
        index.append(-1)    # 分类标签附加在最后
        colName = stdTrainFrame.columns[index]  # 获取对应索引值的列标签
    elif method == 'var':    # 以类间方差最大为标准提取主要特征
        # 获取male各项特征均值列表
        maleSampleMean = list(stdTrainFrame[stdTrainFrame.iloc[:, -1] == 1].mean(axis=0).iloc[0:-1])
        # 获取female各项特征均值列表
        femaleSampleMean = list(stdTrainFrame[stdTrainFrame.iloc[:, -1] == 0].mean(axis=0).iloc[0:-1])
        # 对male/female标签计数
        classWeight = np.array(stdTrainFrame.iloc[:, -1].value_counts()).prod() / stdTrainFrame.iloc[:, -1].size
        # 求类间方差列表
        different = [classWeight * (maleSampleMean[i] - femaleSampleMean[i]) ** 2  for i in range(len(maleSampleMean))]
        #print(different)
        # 获取降序特征索引
        index = list(np.argsort(different)[-1:-feature_num - 1:-1])
        index.append(-1)  # 分类标签附加在最后
        colName = stdTrainFrame.columns[index]  # 获取对应索引值的列标签
    elif method == 'boost_weight':  # 以决策树特征权重为标准提取主要特征
        model = XGBClassifier(learning_rate =0.1, n_estimators=1000, max_depth=20, min_child_weight=1, gamma=0,
                          subsample=0.8, colsample_bytree=0.8, objective= 'binary:logistic', nthread=4,
                          scale_pos_weight=1, seed=27)
        model.fit(trainFrame.iloc[:, :-1], trainFrame.iloc[:, -1])
        imp_dict = model.get_booster().get_score(importance_type='weight')
        imp_weight = pd.Series(imp_dict).sort_values(ascending=False) # 特征重要性降序排序
        tmpCol = list(imp_weight.index)  # 获取降序排序后的列标签
        # 将未使用的标签附加在最后
        tmpCol += [tok for tok in colName if not (tok in set(tmpCol))]
        # 取出前feature_num个有效主特征
        colName = tmpCol[:feature_num] + ['label']
    elif method == 'boost_gain':    # 以决策树特征增益为标准提取主要特征
        model = XGBClassifier(learning_rate =0.1, n_estimators=1000, max_depth=20, min_child_weight=1, gamma=0,
                          subsample=0.8, colsample_bytree=0.8, objective= 'binary:logistic', nthread=4,
                          scale_pos_weight=1, seed=27)
        model.fit(trainFrame.iloc[:, :-1], trainFrame.iloc[:, -1])
        imp_dict = model.get_booster().get_score(importance_type='gain')
        imp_gain = pd.Series(imp_dict).sort_values(ascending=False) # 特征重要性降序排序
        tmpCol = list(imp_gain.index)  # 获取降序排序后的列标签
        # 将未使用的标签附加在最后
        tmpCol += [tok for tok in colName if not (tok in set(tmpCol))]
        # 取出前feature_num个有效主特征
        colName = tmpCol[:feature_num] + ['label']
    #print(colName)
    return trainFrame[colName].copy(), testFrame[colName].copy()  # 返回具有主要特征的dataFrame切片
