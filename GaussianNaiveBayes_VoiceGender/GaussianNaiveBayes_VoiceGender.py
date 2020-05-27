import pandas as pd
import numpy as np

from sklearn.model_selection import  train_test_split

def loadDataset(csvFile='voice.csv'):
    dataFrame = pd.read_csv(csvFile)
    #print(dataFrame)
    return dataFrame

def fixDeafults(dataFrame):
    for index, row in dataFrame.iterrows():
        for i in range(len(row)-1):
            if row[i] < 0.000001:
                dataFrame.iloc[index, i] = np.nan
        if row[-1] == 'male': dataFrame.iloc[index, -1] = 1
        else: dataFrame.iloc[index, -1] = 0
    #print(dataFrame)
    #print(dataFrame.mean())
    dataFrame.fillna(dataFrame.mean(), inplace=True)
    #print(dataFrame)
    #print(dataFrame.mean())

def divideDataset(dataFrame, testSize=0.3):
    return train_test_split(dataFrame, test_size=testSize)

class GaussianNaiveBayesClassfier:
    def __init__(self):
        self.prior = None
        self.averages = None
        self.variances = None
        self.n_class = None
    def obtainPrior(self, dataFrame):
        lableCounts = np.array(dataFrame.iloc[:, -1].value_counts()).astype(np.float)
        prior = lableCounts / dataFrame.iloc[:, -1].size
        return prior
    def obtainAverages(self, dataFrame):
        return np.array([dataFrame[dataFrame.iloc[:,-1]==i].mean(axis=0).iloc[0:-1] for i in range(self.n_class)])
    def obtainVariances(self,dataFrame):
        return np.array([dataFrame[dataFrame.iloc[:, -1] == i].var(axis=0).iloc[0:-1] for i in range(self.n_class)])
    def obtainProbDensity(self, vector):
        density = np.exp(-(vector - self.averages) ** 2 / (2 * self.variances)) / np.sqrt(self.variances)
        return density.sum(axis=1)
    def fit(self, trainFrame):
        self.prior = self.obtainPrior(trainFrame)
        self.n_class = len(self.prior)
        self.averages = self.obtainAverages(trainFrame)
        self.variances = self.obtainVariances(trainFrame)
    def predictProb(self, testFrame):
        probDensity = np.apply_along_axis(self.obtainProbDensity, axis=1, arr=testFrame.values)
        jointProb = self.prior * probDensity
        jointSum = jointProb.sum(axis=1)
        #print(jointProb.shape,jointSum.shape)
        return jointProb / jointSum[:,None]
    def predict(self, testFrame):
        return self.predictProb(testFrame).argmax(axis=1)

def main():
    df = loadDataset()
    fixDeafults(df)
    #print(df)
    trainFrame, testFrame = divideDataset(df)
    #print(trainFrame)
    clf = GaussianNaiveBayesClassfier()
    clf.fit(trainFrame)
    voicePredict = clf.predict(testFrame.iloc[:,0:-1])
    voiceRef = testFrame.iloc[:,-1].values
    #print(voicePredict)
    maleNum = sum(voiceRef==1)
    femaleNum = sum(voiceRef==0)
    diff = voicePredict - voiceRef
    maleFail = sum(diff==-1)
    femaleFail = sum(diff==1)
    totFail = maleFail + femaleFail
    maleFailRate = float(maleFail) / float(maleNum)
    femaleFailRate = float(femaleFail) / float(femaleNum)
    totFailRate = float(totFail) / float(diff.size)
    print('男声正确率：%.4f%%' % (100.0 - maleFailRate * 100))
    print('男声错误率: %.4f%%' % (maleFailRate * 100))
    print('女声正确率：%.4f%%' % (100.0 - femaleFailRate * 100))
    print('女声错误率: %.4f%%' % (femaleFailRate * 100))
    print('总体正确率：%.4f%%' % (100.0 - totFailRate * 100))

if __name__ == '__main__':
    main()
    main()
    main()


"""
df = loadDataset()
#print(df)
fixDeafults(df)
#print(df)
#trainFrame, testFrame = divideDataset(df)
#print(divideDataset(df))
print(df.iloc[:,-1].size)
print(np.array(df.iloc[:,-1].value_counts()).astype(np.float) / df.values.shape[0])
print(np.array([df[df.iloc[:,-1]==i].mean(axis=0).iloc[0:-1] for i in range(2)]))
"""