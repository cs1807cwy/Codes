#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from PIL import Image
from pylab import *
import operator
# 导入线性代数库
import numpy as np
# 导入画图工具
import matplotlib.pyplot as plt

def percepton():
    #感知机演示
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
    predict(X, y)   #调用预测函数

def predict(trainMat, trainLabel):  #预测函数
    w = np.array([0, 1, 0])
    step = 0
    modify = 0
    while True:
        mis = 0
        for i in range(0, trainMat.shape[0]):
            plt.clf()
            step += 1
            x = np.r_[trainMat[i], np.array([1])]
            y = trainLabel[i]
            delta = 0
            for ind in range(0, w.shape[0]):
                delta += y * w[ind] * x[ind]    #判别分类结果
            if delta <= 0:
                if w[1] == 0:
                    Dotx = [-w[2]/w[0]]
                    Doty = [-5, 10]
                else:
                    Dotx = [-5, 10]
                    Doty = [-5 * (-w[0] / w[1]) - w[2] / w[1], 10 * (-w[0] / w[1]) - w[2] / w[1]]
                #单步更新显示
                plt.plot(Dotx, Doty)
                plt.scatter(trainMat[:,0], trainMat[:,1], c=trainLabel, cmap=plt.cm.spring, edgecolor="k")
                plt.scatter(trainMat[i][0], trainMat[i][1], marker='*', c='red', s=200)
                plt.title("Classifier:Percepton, Step:%d, Modify:%d\nPress Enter to continue..." % (step, modify))
                plt.xlim(-5, 10)
                plt.ylim(-5, 10)
                ginput(0)
                modify += 1
                w = w + y * x
                mis += 1    #错误计数
        if mis == 0:
            break
    print("w =", w)
    for i in range(0, trainMat.shape[0]):
        x = np.r_[trainMat[i], np.array([1])]
        y = trainLabel[i]
        print("X =", x)
        print("y =", y)
        delta = 0
        for ind in range(0, w.shape[0]):
            delta += y * w[ind] * x[ind]
        print("delta =", delta)
    Dotx = [-5, 10]
    Doty = [-5 * (-w[0] / w[1]) - w[2] / w[1], 10 * (-w[0] / w[1]) - w[2] / w[1]]
    plt.plot(Dotx, Doty)
    plt.scatter(trainMat[:, 0], trainMat[:, 1], c=trainLabel, cmap=plt.cm.spring, edgecolor="k")
    plt.title("Classifier:Percepton, Step:%d, Modify:%d\nEnds!" % (step, modify))
    plt.xlim(-5, 10)
    plt.ylim(-5, 10)
    ginput(0)
    ginput(0)

percepton()
