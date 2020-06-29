#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt

class Performance:
    """
    定义一个类，用来分类器的性能度量
    """
    def __init__(self, labels, scores, threshold=0.5):
        """
        :param labels:数组类型，真实的标签
        :param scores:数组类型，分类器的得分
        :param threshold:检测阈值
        """
        self.labels = labels
        self.scores = scores
        self.threshold = threshold
        self.db = self.get_db()
        self.tp, self.fp, self.fn, self.tn = self.get_confusion_matrix()    # int型
        self.TP, self.FP, self.FN, self.TN = float(self.tp), float(self.fp), float(self.fn), float(self.tn) # float型

    def set(self, tp_fp_fn_tn):
        self.tp, self.fp, self.fn, self.tn = tp_fp_fn_tn
        self.TP, self.FP, self.FN, self.TN = float(self.tp), float(self.fp), float(self.fn), float(self.tn)
    def Prevalence(self):
        """
        :return: 阳性率
        """
        return (self.TP + self.FN) / (self.TP + self.FN + self.FP + self.TN)

    def ACC(self):
        """
        :return: 准确度
        """
        return (self.TP + self.TN) / (self.TP + self.FN + self.FP + self.TN)

    def PPV(self):
        """
        :return: 阳性预测值
        """
        return self.TP / (self.TP + self.FP)

    def FDR(self):
        """
        :return: 错误发现率
        """
        return self.FP / (self.TP + self.FP)

    def FOR(self):
        """
        :return: 错误遗漏率
        """
        return self.FN / (self.TN + self.FN)

    def NPV(self):
        """
        :return: 阴性预测值
        """
        return self.TN / (self.TN + self.FN)

    def TPR(self):
        """
        :return: 真阳性率
        """
        return self.TP / (self.TP + self.FN)

    def FNR(self):
        """
        :return: 假阴性率
        """
        return self.FN / (self.TP + self.FN)

    def FPR(self):
        """
        :return: 假阳性率
        """
        return self.FP / (self.FP + self.TN)

    def TNR(self):
        """
        :return: 真阴性率
        """
        return self.TN / (self.FP + self.TN)

    def PLR(self):
        """
        :return: 阳性似然比
        """
        return self.TPR() / self.FPR()

    def NLR(self):
        """
        :return: 阳性似然比
        """
        return self.FNR() / self.TNR()

    def DOR(self):
        """
        :return: 诊断比值比
        """
        return self.PLR() / self.NLR()

    def F_score(self, beta=1.0):
        """
        :return: F-beta分数
        """
        return (1 + beta ** 2) * self.PPV() * self.TPR() / (beta ** 2 * self.PPV() + self.TPR())

    def AUC(self):
        """
        :return: ROC曲线下的面积
        """
        auc = 0.
        prev_x = 0
        xy_arr = self.roc_coord()
        for x, y in xy_arr:
            if x != prev_x:
                auc += (x - prev_x) * y
                prev_x = x
        return auc

    def roc_coord(self):
        """
        :return: roc坐标
        """
        xy_arr = []
        tp, fp = 0., 0.
        neg = self.TN + self.FP
        pos = self.TP + self.FN
        for i in range(len(self.db)):
            tp += self.db[i][0]
            fp += 1 - self.db[i][0]
            xy_arr.append([fp / neg, tp / pos])
        return xy_arr

    def roc_plot(self):
        """
        画roc曲线
        """
        auc = self.AUC()
        xy_arr = self.roc_coord()
        x = [_v[0] for _v in xy_arr]
        y = [_v[1] for _v in xy_arr]
        plt.title("ROC curve (AUC = %.4f)" % auc)
        plt.ylabel("True Positive Rate")
        plt.xlabel("False Positive Rate")
        plt.plot(x, y)
        plt.show()

    def get_db(self):
        db = []
        for i in range(len(self.labels)):
            db.append([self.labels[i], self.scores[i]])
        db = sorted(db, key=lambda x: x[1], reverse=True)     # 获取降序排列的阈值
        return db

    def get_confusion_matrix(self):
        """
        计算混淆矩阵
        :return:
        """
        tp, fp, fn, tn = 0, 0, 0, 0
        for i in range(len(self.labels)):
            if self.labels[i] == 1 and self.scores[i] >= self.threshold:
                tp += 1 # 真阳例
            elif self.labels[i] == 0 and self.scores[i] >= self.threshold:
                fp += 1 # 假阳例
            elif self.labels[i] == 1 and self.scores[i] < self.threshold:
                fn += 1 # 假阴例
            else:
                tn += 1 # 真阴例
        return [tp, fp, fn, tn]
