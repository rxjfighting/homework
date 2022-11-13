# coding=utf-8
"""
数据集：Mnist
训练集数量：60000(实际使用：1000)
测试集数量：10000（实际使用：100)
------------------------------
运行结果：
    正确率：99%
"""

import time
import numpy as np
import math
import random


def loadData(fileName):
    dataArr = []
    labelArr = []
    fr = open(fileName)
    for line in fr.readlines():
        curLine = line.strip().split(",")
        dataArr.append([int(num) / 255 for num in curLine[1:]])
        if int(curLine[0]) == 0:
            labelArr.append(1)
        else:
            labelArr.append(-1)
    return dataArr, labelArr


class SVM:
    def __init__(self, trainDataList, trainLabelList, sigma=10, C=200, toler=0.001):

        self.trainDataMat = np.mat(trainDataList)  # 训练数据集
        self.trainLabelMat = np.mat(trainLabelList).T  # 训练标签集，为了方便后续运算提前做了转置，变为列向量
        self.m, self.n = np.shape(self.trainDataMat)  # m：训练集数量    n：样本特征数目
        self.sigma = sigma  # 高斯核分母中的σ
        self.C = C  # 惩罚参数
        self.toler = toler  # 松弛变量
        self.k = self.calcKernel()  # 核函数（初始化时提前计算）
        self.b = 0  # SVM中的偏置b
        self.alpha = [0] * self.trainDataMat.shape[0]  # α为待求的参数，长度为训练集数目
        self.E = [0 * self.trainLabelMat[i, 0] for i in range(self.trainLabelMat.shape[0])]  # SMO运算过程中的Ei，初始化为0
        self.supportVecIndex = []  # 支持向量在训练集中的位置

    def calcKernel(self):
        """
        计算核函数,使用的是高斯核 详见“7.3.3 常用核函数” 式 7.90
        :return: 高斯核矩阵
        """
        # 初始化高斯核结果矩阵 大小 = 训练集长度m * 训练集长度m，元素初始值=0
        # k[i][j] = Xi * Xj
        k = [[0 for i in range(self.m)] for j in range(self.m)]

        # 大循环遍历Xi，Xi为式7.90中的x
        for i in range(self.m):
            # 每100个打印一次
            if i % 100 == 0:
                print('construct the kernel:', i, self.m)
            # 得到式7.90中的X
            X = self.trainDataMat[i, :]
            # 小循环遍历Xj，Xj为式7.90中的Z
            # 由于 Xi * Xj 等于 Xj * Xi，一次计算得到的结果可以
            # 同时放在k[i][j]和k[j][i]中，这样一个矩阵只需要计算一半即可
            # 所以小循环从i开始
            for j in range(i, self.m):
                # 获得Z
                Z = self.trainDataMat[j, :]
                # 先计算||X - Z||^2
                result = (X - Z) * (X - Z).T
                # 分子除以分母后，计算指数函数的值，得到的即为高斯核结果
                result = np.exp(-1 * result / (2 * self.sigma ** 2))
                # 将Xi*Xj的结果存放入k[i][j]和k[j][i]中，因该矩阵显然是一个对称矩阵
                k[i][j] = result
                k[j][i] = result
        # 返回高斯核矩阵
        return k

    def calcSinglKernel(self, x1, x2):
        """
        单独计算核函数，为预测的时候用
        :param x1:向量1
        :param x2: 向量2
        :return: 核函数结果
        """
        # 按照“7.3.3 常用核函数”式7.90计算高斯核
        result = (x1 - x2) * (x1 - x2).T
        result = np.exp(-1 * result / (2 * self.sigma ** 2))
        # 返回结果
        return np.exp(result)

    def calc_gxi(self, i):
        gxi = 0
        index = [i for i, alpha in enumerate(self.alpha) if alpha != 0]
        for j in index:
            gxi += self.alpha[j] * self.trainLabelMat[j] * self.k[j][i]
        gxi += self.b
        return gxi

    def isSatisfyKKT(self, i):
        gxi =self.calc_gxi(i)
        yi = self.trainLabelMat[i]
        if(math.fabs(self.alpha[i]) < self.toler) and (yi * gxi >=1):
            return True
        elif (math.fabs(self.alpha[i] - self.C) < self.toler) and (yi * gxi <= 1):
            return True
        elif(self.alpha[i] > -self.toler) and (self.alpha[i] < (self.C + self.toler)) \
            and (math.fabs(yi * gxi - 1) < self.toler):
            return True
        return False

    def calcEi(self, i):
        gxi = self.calc_gxi(i)
        return gxi - self.trainLabelMat[i]

    def getAlpha2(self, E1, i):
        E2 = 0
        maxE1_E2 = -1
        maxIndex = -1
        nozeroE = [i for i, Ei in enumerate(self.E) if Ei != 0]
        for j in nozeroE:
            E2_tmp = self.calcEi(j)
            if math.fabs(E1 - E2_tmp) > maxE1_E2:
                maxE1_E2 = math.fabs(E1 - E2_tmp)
                E2 = E2_tmp
                maxIndex = j
        if maxIndex == -1:
            maxIndex = i
        while maxIndex == i:
            maxIndex = int(random.uniform(0, self.m))
        E2 = self.calcEi(maxIndex)
        return E2, maxIndex

    def train(self, iter=100):
        iterStep = 0
        parameterChanged = 1
        while (iterStep < iter) and (parameterChanged > 0):
            print('iter:%d:%d' % (iterStep, iter))
            iterStep += 1
            parameterChanged = 0
            for i in range(self.m):
                if self.isSatisfyKKT(i) == False:
                    E1 = self.calcEi(i)
                    E2, j = self.getAlpha2(E1, i)
                    y1 = self.trainLabelMat[i]
                    y2 = self.trainLabelMat[j]
                    alphaold_1 = self.alpha[i]
                    alphaold_2 = self.alpha[j]
                    if y1 != y2:
                        L = max(0, alphaold_2 - alphaold_1)
                        H = min(self.C, self.C + alphaold_2 - alphaold_1)
                    else:
                        L  = max(0, alphaold_2 + alphaold_1 - self.C)
                        H = min(self.C, alphaold_2 + alphaold_1)
                    if L == H:
                        continue
                    k11 = self.k[i][i]
                    k22 = self.k[j][j]
                    k21 = self.k[j][i]
                    k12 = self.k[i][j]
                    alphaNew_2 = alphaold_2 + y2 * (E1 - E2) / (k11 +k22 - 2*k12)
                    if alphaNew_2 < L:
                        alphaNew_2 = L
                    elif alphaNew_2 > H:
                        alphaNew_2 = H
                    alphaNew_1 = alphaold_1 + y1 * y2 * (alphaold_2 - alphaNew_2)
                    b1New = -1 * E1 - y1 * k11 * (alphaNew_1 - alphaold_1) \
                        - y2 * k21 * (alphaNew_2 - alphaold_2) + self.b
                    b2New = -1 * E2 - y1 * k12 * (alphaNew_1 - alphaold_1) \
                        - y2 * k22 * (alphaNew_2 - alphaold_2) + self.b
                    if (alphaNew_1 > 0 ) and (alphaNew_1 < self.C):
                        bNew = b1New
                    elif (alphaNew_2 > 0 ) and (alphaNew_2 < self.C):
                        bNew = b2New
                    else:
                        bNew = (b1New + b2New) / 2
                    self.alpha[i] = alphaNew_1
                    self.alpha[j] = alphaNew_2
                    self.b = bNew
                    self.E[i] = self.calcEi(i)
                    self.E[j] = self.calcEi(j)
                    if math.fabs(alphaNew_2 - alphaold_2) >= 0.00001:
                        parameterChanged += 1
                print('iter: %d i:%d, pairs changed %d' % (iterStep, i, parameterChanged))
        for i in range(self.m):
            if self.alpha[i] > 0:
                self.supportVecIndex.append(i)

    def predict(self, x):
        result = 0
        for i in self.supportVecIndex:
            tmp = self.calcSinglKernel(self.trainDataMat[i, :], np.mat(x))
            result += self.alpha[i] * self.trainLabelMat[i] * tmp
        result += self.b
        return np.sign(result)

    def test(self, testDataList, testLabelList):
        errorCnt = 0
        for i in range(len(testDataList)):
            print('test:%d:%d' % (i, len(testDataList)))
            result = self.predict((testDataList[i]))
            if result != testLabelList[i]:
                errorCnt +=1
        return 1 - errorCnt / len(testDataList)


if __name__ == '__main__':
    start = time.time()
    print('start read transSet')
    trainDataList, trainLabelList = loadData('mnist_train.csv')
    print('start read testSet')
    testDataList, testLabelList = loadData('mnist_test.csv')
    print('start init SVM')
    svm = SVM(trainDataList[:1000], trainLabelList[:1000], 10, 200, 0.001)
    print('start to train')
    svm.train()
    print('start to test')
    accuracy = svm.test(testDataList[:100], testLabelList[:100])
    print('the accuracy is:%d' % (accuracy * 100), '%')
    print('time span', time.time() - start)
