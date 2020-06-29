#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
from pylab import MaxNLocator
from VoiceFeaturePretreating import *
from GaussianNaiveBayesClassifier import GaussianNaiveBayesClassfier
from Measure import Performance

"""
函数说明:统计函数
Parameters:
    trainFrames - 训练集列表
    testFrames - 测试集列表
    denoise - 动态降噪 default:False
    bagging - 附加集成分类器数量 default:0
    weakness - 附加集成分类器强度 default:0.3
Returns:
    (denoise, bagging, accuracy, feature, matrix, prevalence, max_acc, max_fea, min_acc, min_fea, acc, var, ppv, npv, tpr, tnr, plr, nlr, dor, f1) -
    (动态降噪, 集成数量, 准确度列表, 特征列表, 总混淆矩阵, 总男声比例, 最大准确度, 最大准确度使用特征, 最低准确度, 最低准确度使用特征, 平均准确度,
    准确度方差, 平均男声查准率, 平均女声查准率, 平均男声查全率, 平均女声查全率, 平均男声似然比, 平均女声似然比, 平均诊断比值比, 平均F1分数)
"""
def statistic(trainFrames, testFrames, denoise=False, bagging=0, weakness=0.3, detail_show=False, roc_plot=False, global_plot=False):
    accuracy = []   # 准确度列表
    feature = []    # 特征列表
    confusionBase = np.array([0, 0, 0, 0])  # 混淆矩阵
    for i in range(len(trainFrames)):
        clf = GaussianNaiveBayesClassfier(has_denoise=denoise, bagging_rate=bagging, bagging_weakness=weakness)
        clf.fit(trainFrames[i])
        voiceProb = clf.predict(testFrames[i].iloc[:,0:-1], 'prob')
        #voicePredict = voiceProb.argmax(axis=1)
        label = testFrames[i].loc[:, 'label']
        #print(label)
        p = Performance(list(label), list(voiceProb[:, 1]))
        confusion = p.get_confusion_matrix()
        confusionBase = confusionBase + np.array(confusion)
        #print(confusion)
        acc = p.ACC()
        accuracy.append(acc)
        fea = trainFrames[i].columns.values.tolist()
        feature.append(fea)
        if detail_show: # 单次统计数据
            prevalence = p.Prevalence()
            ppv = p.PPV()
            npv = p.NPV()
            tpr = p.TPR()
            tnr = p.TNR()
            plr = p.PLR()
            nlr = p.NLR()
            dor = p.DOR()
            f1 = p.F_score(1.0)
            #print((confusion))
            matDict = {'男声': [int(confusion[0]), int(confusion[2])],
                       '女声': [int(confusion[1]), int(confusion[3])]}
            matrix = pd.DataFrame(matDict, index=['预测男声', '预测女声'])
            print('\n------------第%d次测试------------' % (i+1))
            print('特征：', fea)
            print('动态降噪：', denoise)
            print('集成：', bagging)
            print('混淆矩阵：')
            print(matrix)
            print('总男声比例：%.2f%%' % (prevalence * 100))
            print('总体准确度：%.2f%%' % (acc * 100))
            print('男声查准率：%.2f%%' % (ppv * 100))
            print('男声查全率: %.2f%%' % (tpr * 100))
            print('女声查准率：%.2f%%' % (npv * 100))
            print('女声查全率: %.2f%%' % (tnr * 100))
            print('男声似然比：%.4f' % plr)
            print('女声似然比：%.4f' % (1 / nlr))
            print('判别男声相关：')
            print('诊断比值比：%.4f' % dor)
            print('F1分数：%.4f' % f1)
        if roc_plot:    # 以男声为阳例的ROC曲线
            p.roc_plot()
    # 总体统计数据
    res = Performance([0,1], [0.2,0.8])
    res.set(list(confusionBase))
    prevalence = res.Prevalence()
    acc = res.ACC()
    ppv = res.PPV()
    npv = res.NPV()
    tpr = res.TPR()
    tnr = res.TNR()
    plr = res.PLR()
    nlr = res.NLR()
    dor = res.DOR()
    f1 = res.F_score(1.0)
    max_acc = max(accuracy)
    max_fea = feature[accuracy.index(max_acc)]
    min_acc = min(accuracy)
    min_fea = feature[accuracy.index(min_acc)]
    var = np.var(accuracy)
    #print(confusionBase)
    matDict = {'男声':[int(confusionBase[0]), int(confusionBase[2])],
              '女声':[int(confusionBase[1]), int(confusionBase[3])]}
    matrix = pd.DataFrame(matDict, index=['预测男声','预测女声'])
    print('\n\n------------测试总次数=%d------------' % len(trainFrames))
    print('动态降噪：', denoise)
    print('集成：', bagging)
    print('累积混淆矩阵：')
    print(matrix)
    print('总男声比例：%.2f%%' % (prevalence * 100))
    print('最大准确度：%.2f%%' % (max_acc * 100))
    print('最大准确度使用特征：', max_fea)
    print('最低准确度：%.2f%%' % (min_acc * 100))
    print('最低准确度使用特征：', min_fea)
    print('平均准确度：%.2f%%' % (acc * 100))
    print('准确度方差：%g' % var)
    print('平均男声查准率：%.2f%%' % (ppv * 100))
    print('平均男声查全率：%.2f%%' % (tpr * 100))
    print('平均女声查准率：%.2f%%' % (npv * 100))
    print('平均女声查全率：%.2f%%' % (tnr * 100))
    print('平均男声似然比：%.4f' % plr)
    print('平均女声似然比：%.4f' % (1 / nlr))
    print('判别男声相关：')
    print('平均诊断比值比：%.4f' % dor)
    print('平均F1分数：%.4f' % f1)
    if global_plot: # 显示各次试验的准确率图
        ax = plt.figure().gca()
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        plt.xticks(np.arange(1, len(accuracy) + 1, 1))
        #plt.title("Accuracy for %d turns" % (len(accuracy)))
        plt.title("Accuracy for %d turns(denoise=%d,bagging=%d)" % (len(accuracy), denoise, bagging))
        plt.ylabel("Accuracy")
        plt.xlabel("turns")
        if len(accuracy) > 1:
            plt.plot(list(range(1, len(accuracy) + 1)), accuracy)
        else:
            plt.scatter(1, accuracy, c=2, cmap=plt.cm.spring, edgecolors='k')
        plt.show()
    return denoise, bagging, accuracy, feature, matrix, prevalence, max_acc, max_fea, min_acc, min_fea, acc, var, ppv, npv, tpr, tnr, plr, nlr, dor, f1

"""
函数说明:参数测试
Parameters:
    seed - 训练集/测试集种子
"""
def main_cross(seed=1, detail_show=False, roc_plot=False, global_plot=False):
    df_train = loadDataset('voice/voice_train_%d.csv' % seed)
    df_test = loadDataset('voice/voice_test_%d.csv' % seed)
    fixDeafults(df_train, discard=True)
    fixDeafults(df_test, discard=False)
    #print(df_train)
    df_mean_closet = [generateMainFeatures(df_train, df_test, i, 'mean') for i in range(1, df_train.shape[1])]
    df_var_closet = [generateMainFeatures(df_train, df_test, i, 'var') for i in range(1, df_train.shape[1])]
    df_boostw_closet = [generateMainFeatures(df_train, df_test, i, 'boost_weight') for i in range(1, df_train.shape[1])]
    df_boostg_closet = [generateMainFeatures(df_train, df_test, i, 'boost_gain') for i in range(1, df_train.shape[1])]
    mean_train = [tup[0] for tup in df_mean_closet]
    mean_test = [tup[1] for tup in df_mean_closet]
    var_train = [tup[0] for tup in df_var_closet]
    var_test = [tup[1] for tup in df_var_closet]
    w_train = [tup[0] for tup in df_boostw_closet]
    w_test = [tup[1] for tup in df_boostw_closet]
    g_train = [tup[0] for tup in df_boostg_closet]
    g_test = [tup[1] for tup in df_boostg_closet]
    tab = [0, 0, 0, 0, 0, 0, 0, 0]
    # 0:denoise, 1:bagging, 2:accuracy, 3:feature, 4:matrix, 5:prevalence, 6:max_acc, 7:max_fea, 8:min_acc,
    # 9:min_fea, 10:acc, 11:var, 12:ppv, 13:npv, 14:tpr, 15:tnr, 16:plr, 17:nlr, 18:dor, 19:f1
    tab[0] = statistic(mean_train, mean_test, denoise=False, bagging=0, detail_show=detail_show, roc_plot=roc_plot, global_plot=global_plot)
    tab[1] = statistic(var_train, var_test, denoise=False, bagging=0, detail_show=detail_show, roc_plot=roc_plot, global_plot=global_plot)
    tab[2] = statistic(mean_train, mean_test, denoise=True, bagging=0, detail_show=detail_show, roc_plot=roc_plot, global_plot=global_plot)
    tab[3] = statistic(var_train, var_test, denoise=True, bagging=0, detail_show=detail_show, roc_plot=roc_plot, global_plot=global_plot)
    tab[4] = statistic(w_train, w_test, denoise=False, bagging=0, detail_show=detail_show, roc_plot=roc_plot, global_plot=global_plot)
    tab[5] = statistic(g_train, g_test, denoise=False, bagging=0, detail_show=detail_show, roc_plot=roc_plot, global_plot=global_plot)
    tab[6] = statistic(w_train, w_test, denoise=True, bagging=0, detail_show=detail_show, roc_plot=roc_plot, global_plot=global_plot)
    tab[7] = statistic(g_train, g_test, denoise=True, bagging=0, detail_show=detail_show, roc_plot=roc_plot, global_plot=global_plot)
    print('------------模式准确度列表------------')
    print('In mode: denoise = False, method = \'mean\'')
    print('模式最大准确度：%.2f%%' % (100 * tab[0][6]))
    print('with', tab[0][7], ', denoise = False , method = \'mean\'')
    print('模式最低准确度：%.2f%%' % (100 * tab[0][8]))
    print('with', tab[0][9], ', denoise = False , method = \'mean\'')
    print('模式平均准确度：%.2f%%' % (100 * tab[0][10]))
    print('模式准确度方差：%g' % (tab[0][11]))
    print('-------------------------------------')
    print('In mode: denoise = False, method = \'var\'')
    print('模式最大准确度：%.2f%%' % (100 * tab[1][6]))
    print('with', tab[1][7], ', denoise = False , method = \'var\'')
    print('模式最低准确度：%.2f%%' % (100 * tab[1][8]))
    print('with', tab[1][9], ', denoise = False , method = \'var\'')
    print('模式平均准确度：%.2f%%' % (100 * tab[1][10]))
    print('模式准确度方差：%g' % (tab[1][11]))
    print('-------------------------------------')
    print('In mode: denoise = True, method = \'mean\'')
    print('模式最大准确度：%.2f%%' % (100 * tab[2][6]))
    print('with', tab[2][7], ', denoise = True , method = \'mean\'')
    print('模式最低准确度：%.2f%%' % (100 * tab[2][8]))
    print('with', tab[2][9], ', denoise = True , method = \'mean\'')
    print('模式平均准确度：%.2f%%' % (100 * tab[2][10]))
    print('模式准确度方差：%g' % (tab[2][11]))
    print('-------------------------------------')
    print('In mode: denoise = True, method = \'var\'')
    print('模式最大准确度：%.2f%%' % (100 * tab[3][6]))
    print('with', tab[3][7], ', denoise = True , method = \'var\'')
    print('模式最低准确度：%.2f%%' % (100 * tab[3][8]))
    print('with', tab[3][9], ', denoise = True , method = \'var\'')
    print('模式平均准确度：%.2f%%' % (100 * tab[3][10]))
    print('模式准确度方差：%g' % (tab[3][11]))
    print('-------------------------------------')
    print('In mode: denoise = False, method = \'boost_weight\'')
    print('模式最大准确度：%.2f%%' % (100 * tab[4][6]))
    print('with', tab[4][7], ', denoise = False , method = \'boost_weight\'')
    print('模式最低准确度：%.2f%%' % (100 * tab[4][8]))
    print('with', tab[4][9], ', denoise = False , method = \'boost_weight\'')
    print('模式平均准确度：%.2f%%' % (100 * tab[4][10]))
    print('模式准确度方差：%g' % (tab[4][11]))
    print('-------------------------------------')
    print('In mode: denoise = False, method = \'boost_gain\'')
    print('模式最大准确度：%.2f%%' % (100 * tab[5][6]))
    print('with', tab[5][7], ', denoise = False , method = \'boost_gain\'')
    print('模式最低准确度：%.2f%%' % (100 * tab[5][8]))
    print('with', tab[5][9], ', denoise = False , method = \'boost_gain\'')
    print('模式平均准确度：%.2f%%' % (100 * tab[5][10]))
    print('模式准确度方差：%g' % (tab[5][11]))
    print('-------------------------------------')
    print('In mode: denoise = True, method = \'boost_weight\'')
    print('模式最大准确度：%.2f%%' % (100 * tab[6][6]))
    print('with', tab[6][7], ', denoise = True , method = \'boost_weight\'')
    print('模式最低准确度：%.2f%%' % (100 * tab[6][8]))
    print('with', tab[6][9], ', denoise = True , method = \'boost_weight\'')
    print('模式平均准确度：%.2f%%' % (100 * tab[6][10]))
    print('模式准确度方差：%g' % (tab[6][11]))
    print('-------------------------------------')
    print('In mode: denoise = True, method = \'boost_gain\'')
    print('模式最大准确度：%.2f%%' % (100 * tab[7][6]))
    print('with', tab[7][7], ', denoise = True , method = \'boost_gain\'')
    print('模式最低准确度：%.2f%%' % (100 * tab[7][8]))
    print('with', tab[7][9], ', denoise = True , method = \'boost_gain\'')
    print('模式平均准确度：%.2f%%' % (100 * tab[7][10]))
    print('模式准确度方差：%g' % (tab[7][11]))
    print('-------------------------------------')
    print('最大准确度：%.2f%%' % (100 * max([tup[6] for tup in tab])))
    # 0:denoise, 1:bagging, 2:accuracy, 3:feature, 4:matrix, 5:prevalence, 6:max_acc, 7:max_fea, 8:min_acc,
    # 9:min_fea, 10:acc, 11:var, 12:ppv, 13:npv, 14:tpr, 15:tnr, 16:plr, 17:nlr, 18:dor, 19:f1
    plt.style.use('ggplot')
    labels = ['(Mean)', '(Var)', '(Mean,Denoise)', '(Var,Denoise)', '(Weight)',
              '(Gain)', '(Weight,Denoise)', '(Gain,Denoise)']
    # 平均准确度制图
    cont = [tup[10] for tup in tab]
    # 绘制条形图
    plt.bar(x=range(len(tab)),  # 指定条形图x轴的刻度值
            height=cont,  # 指定条形图y轴的数值
            tick_label=labels,  # 指定条形图x轴的刻度标签
            color='steelblue',  # 指定条形图的填充色
            width=0.4
            )
    # 添加y轴的标签
    plt.ylabel('Average Accuracy')
    # 添加条形图的标题
    plt.title('Extraction & Denoise - Average Accuracy Relationship')
    # 为每个条形图添加数值标签
    for i in range(len(cont)):
        plt.text(i, cont[i], '%.4f' % cont[i], ha='center')
    # 显示图形
    plt.show()
    # 最大准确度制图
    cont = [tup[6] for tup in tab]
    plt.bar(x=range(len(tab)), height=cont, tick_label=labels, color='steelblue', width=0.4)
    plt.ylabel('Max Accuracy')
    plt.title('Extraction & Denoise - Max Accuracy Relationship')
    for i in range(len(cont)):
        plt.text(i, cont[i], '%.4f' % cont[i], ha='center')
    plt.show()
    # 最低准确度制图
    cont = [tup[8] for tup in tab]
    plt.bar(x=range(len(tab)), height=cont, tick_label=labels, color='steelblue', width=0.4)
    plt.ylabel('Min Accuracy')
    plt.title('Extraction & Denoise - Min Accuracy Relationship')
    for i in range(len(cont)):
        plt.text(i, cont[i], '%.4f' % cont[i], ha='center')
    plt.show()
    # 0:denoise, 1:bagging, 2:accuracy, 3:feature, 4:matrix, 5:prevalence, 6:max_acc, 7:max_fea, 8:min_acc,
    # 9:min_fea, 10:acc, 11:var, 12:ppv, 13:npv, 14:tpr, 15:tnr, 16:plr, 17:nlr, 18:dor, 19:f1
    # 准确度方差制图
    cont = [tup[11] for tup in tab]
    plt.bar(x=range(len(tab)), height=cont, tick_label=labels, color='steelblue', width=0.4)
    plt.ylabel('Accuracy Variance')
    plt.title('Extraction & Denoise - Accuracy Variance Relationship')
    for i in range(len(cont)):
        plt.text(i, cont[i], '%.6f' % cont[i], ha='center')
    plt.show()
    # 平均男声查准率制图
    cont = [tup[12] for tup in tab]
    plt.bar(x=range(len(tab)), height=cont, tick_label=labels, color='steelblue', width=0.4)
    plt.ylabel('Average Precision')
    plt.title('Extraction & Denoise - Average Male Precision Relationship')
    for i in range(len(cont)):
        plt.text(i, cont[i], '%.4f' % cont[i], ha='center')
    plt.show()
    # 平均男声查全率制图
    cont = [tup[14] for tup in tab]
    plt.bar(x=range(len(tab)), height=cont, tick_label=labels, color='steelblue', width=0.4)
    plt.ylabel('Average Recall')
    plt.title('Extraction & Denoise - Average Male Recall Relationship')
    for i in range(len(cont)):
        plt.text(i, cont[i], '%.4f' % cont[i], ha='center')
    plt.show()
    # 平均女声查准率制图
    cont = [tup[13] for tup in tab]
    plt.bar(x=range(len(tab)), height=cont, tick_label=labels, color='steelblue', width=0.4)
    plt.ylabel('Average Precision')
    plt.title('Extraction & Denoise - Average Female Precision Relationship')
    for i in range(len(cont)):
        plt.text(i, cont[i], '%.4f' % cont[i], ha='center')
    plt.show()
    # 平均女声查全率制图
    cont = [tup[15] for tup in tab]
    plt.bar(x=range(len(tab)), height=cont, tick_label=labels, color='steelblue', width=0.4)
    plt.ylabel('Average Recall')
    plt.title('Extraction & Denoise - Average Female Recall Relationship')
    for i in range(len(cont)):
        plt.text(i, cont[i], '%.4f' % cont[i], ha='center')
    plt.show()
    # 0:denoise, 1:bagging, 2:accuracy, 3:feature, 4:matrix, 5:prevalence, 6:max_acc, 7:max_fea, 8:min_acc,
    # 9:min_fea, 10:acc, 11:var, 12:ppv, 13:npv, 14:tpr, 15:tnr, 16:plr, 17:nlr, 18:dor, 19:f1
    # 男声诊断比值比制图
    cont = [tup[18] for tup in tab]
    plt.bar(x=range(len(tab)), height=cont, tick_label=labels, color='steelblue', width=0.4)
    plt.ylabel('Ratio')
    plt.title('Extraction & Denoise - Male Diagnositc Odds Ratio Relationship')
    for i in range(len(cont)):
        plt.text(i, cont[i], '%.4f' % cont[i], ha='center')
    plt.show()
    # 主要特征提取数量-模式正确率制图
    for i in range(len(tab[0][2])):
        cont = [tup[2][i] for tup in tab]
        plt.bar(x=range(len(tab)), height=cont, tick_label=labels, color='steelblue', width=0.4)
        plt.ylabel('Accuracy')
        plt.title('Extraction & Denoise - with Main Feature(s) %d in Total' % (i + 1))
        for j in range(len(cont)):
            plt.text(j, cont[j], '%.4f' % cont[j], ha='center')
        plt.show()
"""
函数说明:参数确定交叉验证
Parameters:
    tests - 测试轮数 default:10
    features - 提取特征数 default:1
    extraction - 提取特征方法 default:'var'
	    extraction: 'var' - 类间方差最大
	                'mean' - 类均值差最大
	                others - 不提取特征
    denoise - 动态降噪 default:False
    bagging - 附加集成分类器数量 default:0
    weakness - 附加集成分类器强度 default:0.3
"""
def main_in(tests=10, features=1, extraction='var', denoise=False, bagging=0, weakness=0.3, detail_show=False, roc_plot=False, global_plot=True):
    df_train = [loadDataset('voice/voice_train_%d.csv' % i) for i in range(1, tests + 1)]
    df_test = [loadDataset('voice/voice_test_%d.csv' % i) for i in range(1, tests + 1)]
    trainFrames = []
    testFrames = []
    for i in range(len(df_train)):
        fixDeafults(df_train[i], discard=True)
        fixDeafults(df_test[i], discard=False)
        train, test = generateMainFeatures(df_train[i], df_test[i], features, extraction)
        trainFrames.append(train)
        testFrames.append(test)
    ret_val = statistic(trainFrames, testFrames, denoise=denoise, bagging=bagging, weakness=weakness, detail_show=detail_show, roc_plot=roc_plot, global_plot=global_plot)
    print("with:", trainFrames[0].columns.values.tolist(), "denoise =", denoise, ",method =", extraction, ",bagging =", bagging)

"""
函数说明:预测主函数
Parameters:
    train_csv - 训练数据集csv
    test_csv - 测试数据集csv（结果输出目标文件）
    features - 提取特征数 default:1
    extraction - 提取特征方法 default:'var'
	    extraction: 'var' - 类间方差最大
	                'mean' - 类均值差最大
	                others - 不提取特征
    denoise - 动态降噪 default:False
    bagging - 附加集成分类器数量 default:0
    weakness - 附加集成分类器强度 default:0.3
"""
def main_predict(train_csv, test_csv, res_csv, features=1, extraction='var', denoise=False, bagging=0, weakness=0.3):
    df_train = loadDataset(train_csv)
    df_test = loadDataset(test_csv)
    tmp_test = df_test.copy()
    #print('原始测试数据：')
    #print(df_test)
    fixDeafults(df_train, discard=True)
    fixDeafults(tmp_test, discard=False)
    df_train = df_train.loc[:,tmp_test.columns.tolist()]
    #print(df_train)
    train, test = generateMainFeatures(df_train, tmp_test, features, extraction)
    #print(train.columns.values.tolist())
    clf = GaussianNaiveBayesClassfier(has_denoise=denoise, bagging_rate=bagging, bagging_weakness=weakness)
    clf.fit(train)
    #clf.values()
    voiceProb = clf.predict(test.iloc[:, 0:-1], format='prob')
    print('预测概率矩阵：')
    print(voiceProb)
    voicePredict = voiceProb.argmax(axis=1)
    gender = ['female', 'male']
    for i in range(df_test.shape[0]):
        df_test.loc[i, 'label'] = gender[voicePredict[i]]
    df_test.to_csv(res_csv, index=False)
    #print(voicePredict)
    print("with main features:", test.columns[0:-1].values.tolist())
    #for i in range(test.shape[0]):
    #    print(list(test.iloc[i,:-1]))
    print('结果如下:')
    print(df_test)

def myref(seed=1, plt_type='gain'):
    ### load module
    from xgboost import XGBClassifier
    ### load datasets
    if seed > 0:
        df_train = loadDataset('voice/voice_train_%d.csv' % seed)
        df_test = loadDataset('voice/voice_test_%d.csv' % seed)
    else:
        df_train = loadDataset('voice/voice.csv')
        df_test = loadDataset('voice/voice.csv')
    fixDeafults(df_train, discard=True)
    fixDeafults(df_test, discard=False)
    ### fit model for train data
    model = XGBClassifier(learning_rate =0.1, n_estimators=1000, max_depth=20, min_child_weight=1, gamma=0,
                          subsample=0.8, colsample_bytree=0.8, objective= 'binary:logistic', nthread=4,
                          scale_pos_weight=1, seed=27)
    model.fit(df_train.iloc[:,:-1], df_train.loc[:,'label'])
    ### make prediction for test data
    y_pred = model.predict(df_test.iloc[:,:-1])
    y_test = df_test.loc[:,'label'].values
    ### model evaluate
    diff = y_test - y_pred
    acc = float(diff[diff==0].size) / float(diff.size)
    print("accuarcy: %.4f%%" % (acc * 100.0))
    from xgboost import plot_importance
    imp_dict = model.get_booster().get_score(importance_type=plt_type)
    imp = pd.Series(imp_dict).sort_values(ascending=False)
    print(imp)
    fig,ax = plt.subplots(figsize=(10,15))
    plot_importance(model, height=0.5, max_num_features=64, ax=ax, importance_type=plt_type)
    plt.show()

if __name__ == '__main__':
    createDataset(num=10)
    main_cross(1, global_plot=True)
    #main_in(50, 20, None, False, 0)

    #main_in(20, 2, 'mean', False, 0, 0.1, global_plot=False)
    #main_in(20, 20, None, True, 0, 0.1, global_plot=False)
    #main_in(50, 1, 'boost_weight', False, 0)
    #main_in(50, 1, 'boost_gain', False, 0)
    #main_in(50, 2, 'mean', False, 0)
    #main_in(50, 2, 'var', False, 0)
    #main_in(50, 2, 'boost_weight', False, 0)
    #main_in(50, 2, 'boost_gain', False, 0)
    #main_in(50, 3, 'mean', False, 0)
    #main_in(50, 3, 'var', False, 0)
    #main_in(50, 3, 'boost_weight', False, 0)
    #main_in(50, 3, 'boost_gain', False, 0)
    #main_in(50, 4, 'mean', False, 0)
    #main_in(50, 4, 'var', False, 0)
    #main_in(50, 4, 'boost_weight', False, 0)
    #main_in(50, 4, 'boost_gain', False, 0)
    #main_in(50, 5, 'var', False, 0)
    #main_in(50, 6, 'mean', False, 0)

    #main_in(50, 2, 'var', True, 13, 0.1, global_plot=False)
    #main_in(1, 2, 'boost_gain', True, 13, 0.1, roc_plot=True, detail_show=True)
    #main_in(50, 20, None, True, 5, 0.1, global_plot=False)
    #main_in(50, 20, None, True, 5, 0.3, global_plot=False)
    #main_in(50, 20, None, True, 5, 0.5, global_plot=False)
    #main_in(50, 20, None, True, 9, 0.1, global_plot=False)
    #main_in(50, 20, None, True, 9, 0.3, global_plot=False)
    #main_in(50, 20, None, True, 9, 0.5, global_plot=False)
    #main_in(50, 20, None, True, 13, 0.1, global_plot=False)
    #main_in(50, 20, None, True, 13, 0.3, global_plot=False)
    #main_in(50, 20, None, True, 13, 0.5, global_plot=False)

    #main_predict('voice/voice.csv', 'voice/predict/predict.csv', 'voice/predict/res.csv', 20, None, False, 0, 0.3)
    #for i in range(2,20):
    #    main_predict('voice/voice.csv', 'voice/predict/predict.csv', 'voice/predict/res.csv', i, 'var', True, 13, 0.1)
    #myref(0, 'gain')

"""
    df_max_mean_gap =
    df[['IQR', 'mindom', 'meanfun', 'Q25', 'sfm', 'sd', 'maxdom', 'dfrange', 'meandom', 'skew', 'minfun', 'mode', 'meanfreq', 'centroid', 'median', 'modindx', 'sp.ent', 'kurt', 'maxfun', 'Q75', 'label']]
    df_max_between_class_var =
    df[['IQR', 'meanfun', 'sd', 'Q25', 'sp.ent', 'sfm', 'centroid', 'meanfreq', 'median', 'mode', 'meandom', 'maxdom', 'maxfun', 'dfrange', 'mindom', 'minfun', 'Q75', 'skew', 'modindx', 'kurt', 'label']]
    df_boost_weight = 
    df[['meanfun', 'modindx', 'IQR', 'minfun', 'Q25', 'sfm', 'sd', 'mode', 'sp.ent', 'meandom', 'maxdom', 'Q75', 'skew', 'kurt', 'dfrange', 'median', 'meanfreq', 'maxfun', 'mindom', 'centroid', 'label']]
    df_boost_gain = 
    df[['meanfun', 'IQR', 'kurt', 'maxdom', 'sd', 'sfm', 'sp.ent', 'dfrange', 'mindom', 'meanfreq', 'Q75', 'modindx', 'Q25', 'minfun', 'mode', 'meandom', 'skew', 'median', 'maxfun', 'centroid', 'label']]
"""
