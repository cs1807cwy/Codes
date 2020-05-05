#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from PIL import Image
from pylab import *
import operator
# 导入线性代数库
import numpy as np
# 导入画图工具
import matplotlib.pyplot as plt

def knn(dim, k):    #knn演示
    #dim距离维度，k最近邻个数
    plt.imshow(array(Image.new("RGB", (10, 10), (255, 255, 255))), origin='lower')
    print('Please click points')
    plt.title("Press Enter to submit.")
    dotset1 = ginput(0) #接受一类点
    set1X = [dotset1[i][0] for i in range(0, len(dotset1))]
    set1Y = [dotset1[i][1] for i in range(0, len(dotset1))]
    plt.scatter(set1X, set1Y, cmap=plt.cm.spring, edgecolor = "k")
    print('you clicked:', dotset1)
    dotset2 = ginput(0) #接受二类点
    set2X =[dotset2[i][0] for i in range(0, len(dotset2))]
    set2Y = [dotset2[i][1] for i in range(0, len(dotset2))]
    plt.scatter(set2X, set2Y, cmap=plt.cm.spring, edgecolor = "k")
    print('you clicked:', dotset2)
    plt.title("Now have a reference.Press Enter to continue...\nIt may need a few seconds.")
    ginput(0)
    #组合点集
    setX = set1X + set2X
    setY = set1Y + set2Y
    X = np.array([[setX[i], setY[i]] for i in range(0, len(setX))])
    y = np.array([-1 for i in range(0, len(set1X))] + [1 for i in range(0, len(set2X))])
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    #区域显示
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02), np.arange(y_min, y_max, 0.02))
    Z = predict(X, y, np.c_[xx.ravel(), yy.ravel()], dim, k)
    Z = Z.reshape(xx.shape)
    plt.pcolormesh(xx, yy, Z, cmap=plt.cm.Pastel1)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.spring, edgecolor='k')
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.title("Classifier:KNN, k=%d, dim=%d" % (k, dim))
    plt.show()

def predict(trainMat, trainLabel, testMat, dim, k):#pass
    datasetsize = trainMat.shape[0]
    labels = []
    for ind in range(0, testMat.shape[0]):
        inX = testMat[ind]
        # 距离计算，矩阵方式实现
        # 第1步：特征向量求差
        diffMat = np.tile(inX, (datasetsize, 1)) - trainMat
        # 第2步：求距离
        dimenDiffMat = abs(diffMat**dim)
        # 第3步：行序求和，即平方和
        distances = dimenDiffMat.sum(axis=1)
        # 距离从大到小排序，返回距离的序号
        sortedOrders = distances.argsort()
        while k > 0:
            # 构建次数统计表
            classCount = {}
            # 取前k个距离最小的
            for i in range(k):
                # sortedOrders[0]是距离最小的数据样本的序号
                # labels[sortedOrders[0]]是距离最小的数据样本的标签
                votelabel = trainLabel[sortedOrders[i]]
                # 统计标签0-9出现次数
                classCount[votelabel] = classCount.get(votelabel, 0) + 1
            # 排序
            sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
            # 测试是否出现Tie近邻平分状态
            if (len(sortedClassCount) == 1 or sortedClassCount[0][1] != sortedClassCount[1][1]):
                labels.append(sortedClassCount[0][0])
                break
            else: k -= 1
    return np.array(labels)

knn(dim = 1, k=1)