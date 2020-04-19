#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import operator
import struct
#导入线性代数库
import numpy as np
#导入画图工具
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
#训练集文件名
train_images_idx3_ubyte_file = 'train-images.idx3-ubyte'
#训练集标签文件名
train_labels_idx1_ubyte_file = 'train-labels.idx1-ubyte'
#测试集文件名
test_images_idx3_ubyte_file = 't10k-images.idx3-ubyte'
#测试集标签文件名
test_labels_idx1_ubyte_file = 't10k-labels.idx1-ubyte'

#数据集解析函数
def decode_idx3_ubyte(idx3_ubyte_file):
    bin_data = open(idx3_ubyte_file, 'rb').read()
    #解析文件头信息，依次为校验数、图片数、图片高、图片宽
    offset = 0
    fmt_header = '>iiii'
    magic_number, num_images, num_rows, num_cols = struct.unpack_from(fmt_header, bin_data, offset)
    #解析数据集
    image_size = num_rows * num_cols
    offset += struct.calcsize(fmt_header)
    fmt_image = '>' + str(image_size) + 'B'
    images = np.empty((num_images, num_rows, num_cols))
    for i in range(num_images):
        images[i] = np.array(struct.unpack_from(fmt_image, bin_data, offset)).reshape((num_rows, num_cols))
        offset += struct.calcsize(fmt_image)
    return images
#标签集解析函数
def decode_idx1_ubyte(idx1_ubyte_file):
    bin_data = open(idx1_ubyte_file, 'rb').read()
    #解析文件头信息，依次为校验数、标签数
    offset = 0
    fmt_header = '>ii'
    magic_number, num_labels = struct.unpack_from(fmt_header, bin_data, offset)
    #解析标签集
    offset += struct.calcsize(fmt_header)
    fmt_label = '>B'
    labels = np.empty(num_labels)
    for i in range(num_labels):
        labels[i] = struct.unpack_from(fmt_label, bin_data, offset)[0]
        offset += struct.calcsize(fmt_label)
    return labels
    
def classify(inX, dataset, labels, k):
    #inX测试向量，dataset训练矩阵，labels训练标签数组，k最近邻个数
    datasetsize = dataset.shape[0]
    #28*28维距离计算，矩阵方式实现
    #第1步：特征向量求差
    diffMat = np.tile(inX, (datasetsize, 1)) - dataset
    #第2步：求差的平方
    squareDiffMat = diffMat * diffMat
    #第3步：行序求和，即平方和
    squareDistances = squareDiffMat.sum(axis=1)
    #第4步：开根号，获得28*28维特征值距离向量
    distances = squareDistances ** 0.5
    #距离从大到小排序，返回距离的序号
    sortedOrders = distances.argsort()
    while k > 0:
        print("测试近邻数k=%d" % k)
        #构建次数统计表
        classCount = {}
        #取前k个距离最小的
        for i in range(k):
            #sortedOrders[0]是距离最小的数据样本的序号
            #labels[sortedOrders[0]]是距离最小的数据样本的标签
            votelabel = labels[sortedOrders[i]]
            #统计标签0-9出现次数
            classCount[votelabel] = classCount.get(votelabel, 0) + 1
            print(classCount)  #显示统计表
        #排序
        sortedClassCount = sorted(classCount.items(), key = operator.itemgetter(1), reverse = True)
        print(sortedClassCount)  #显示降序统计表
        #测试是否出现Tie近邻平分状态
        if (len(sortedClassCount) == 1 or sortedClassCount[0][1] != sortedClassCount[1][1]):
            return sortedClassCount[0][0]
        k -= 1
        print("降低近邻数重启k=%d" % k)

if __name__ == '__main__':
    train_images = decode_idx3_ubyte(train_images_idx3_ubyte_file)
    train_labels = decode_idx1_ubyte(train_labels_idx1_ubyte_file)
    test_images = decode_idx3_ubyte(test_images_idx3_ubyte_file)
    test_labels = decode_idx1_ubyte(test_labels_idx1_ubyte_file)
    #创建一个读入数据的数组，进行图片特征提取
    m = 60000
    trainingMat = np.zeros((m, 784))  #784 = 28 * 28，这是图片特征向量的维度
    for i in range(m):
        for j in range(28):
            for k in range(28):
                trainingMat[i, 28 * j + k] = train_images[i][j][k]
    errorRateRef = []
    knn = 1
    maxKnn = 13
    while knn <= maxKnn:
        errorCount = 0.0
        mCheck = 10000
        for i in range(mCheck):
            classNumStr = test_labels[i]
            vectorUnderTest = np.zeros(784)
            for j in range(28):
                for k in range(28):
                    vectorUnderTest[28 * j + k] = test_images[i][j][k]  #第i幅测试图
            Result = classify(vectorUnderTest, trainingMat, train_labels, knn)
            print("识别结果: %d 正确答案: %d" % (Result, classNumStr))
            if(Result != classNumStr):
                errorCount += 1.0
                print("错误")
        errorRate = errorCount / float(mCheck)
        print("近邻数k=%d" % knn)
        print("错误数: %d" % errorCount)
        print("错误率: %f" % errorRate)
        print("第%d轮测试结束\n" % ((knn + 1) / 2))
        errorRateRef.append(knn)
        errorRateRef.append(errorRate)
        knn += 1
    print(errorRateRef)
    x = errorRateRef[::2]
    y = errorRateRef[1::2]
    ax = plt.subplot(111)
    xmajorLocator = MultipleLocator(1)
    ax.xaxis.set_major_locator(xmajorLocator)
    ax.xaxis.grid(True, which='major')
    plt.plot(x, y, linewidth=3, color='r',marker='o', markerfacecolor='blue',markersize=12)
    plt.xlabel('KNearestNeighbors') 
    plt.ylabel('ErrorRate') 
    plt.title('ErrorRate-KNearestNeighbors LineGraph')
    plt.show()