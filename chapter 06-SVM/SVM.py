import numpy as np
import time
import torch
from torchvision import datasets,transforms
from PIL import Image


def getData():
    """
    获取数据集:如果本地没有就直接进行下载
    return:train_dataset,test_dataset
    """
    train_dataset=datasets.MNIST(root="./data/"
                                 ,train=True
                                 # ,transform=transforms.ToPILImage()
                                 ,transform=transforms.ToTensor()
                                 ,download=True
                                 )

    test_dataset=datasets.MNIST(root="./data/"
                                ,train=False
                                # , transform=transforms.ToPILImage()
                                ,transform=transforms.ToTensor()
                                ,download=True
                                )
    # print(train_dataset.data,train_dataset.targets.shape,test_dataset.data,test_dataset.targets.shape)
    return train_dataset,test_dataset


# 基于SMO的简化例子
from numpy import *
from time import sleep
def test_SVM():
    train_data,test_data=getData()
    train_label = train_data.targets[0:1000]
    train_data=train_data.data[0:1000]
    # train_label=train_data.label[0:10000]
    test_label = test_data.targets
    test_data=test_data.data
    return train_data,train_label,test_data,test_label



# 参数分别表示第一个alpha的下标，所有alpha的数目
def selectJrand(i,m):
    j=i #we want to select any J not equal to i
    while (j==i):
        j = int(random.uniform(0,m))
    return j

# 调整大于H或或小于L的alpha值
def clipAlpha(aj,H,L):
    if aj > H:
        aj = H
    if L > aj:
        aj = L
    return aj


class optStruct:
    def __init__(self, dataMatIn, classLabels, C, toler):  # Initialize the structure with the parameters
        self.X = dataMatIn
        self.labelMat = classLabels
        self.C = C
        self.tol = toler
        self.m = shape(dataMatIn)[0]
        self.alphas = mat(zeros((self.m, 1)))
        self.b = 0
        # 误差缓存
        self.eCache = mat(zeros((self.m, 2)))  # first column is valid flag


def calcEk(oS, k):
    fXk = float(multiply(oS.alphas, oS.labelMat).T * oS.X * oS.X[k, :].T + oS.b)
    Ek = fXk - float(oS.labelMat[k])
    return Ek


# 内循环中的启发式方法
def selectJ(i, oS, Ei):  # this is the second choice -heurstic, and calcs Ej
    maxK = -1
    maxDeltaE = 0
    Ej = 0
    oS.eCache[i] = [1, Ei]  # set valid #choose the alpha that gives the maximum delta E
    validEcacheList = nonzero(oS.eCache[:, 0].A)[0]
    if (len(validEcacheList)) > 1:
        for k in validEcacheList:  # loop through valid Ecache values and find the one that maximizes delta E
            if k == i:
                continue  # don't calc for i, waste of time
            Ek = calcEk(oS, k)
            deltaE = abs(Ei - Ek)
            if (deltaE > maxDeltaE):
                maxK = k
                maxDeltaE = deltaE
                Ej = Ek
        return maxK, Ej
    else:  # in this case (first time around) we don't have any valid eCache values
        # 选择具有最大步长的j
        j = selectJrand(i, oS.m)
        Ej = calcEk(oS, j)
    return j, Ej


# 计算误差值，并存入缓存中，在优化alpha之后会用到这个值
def updateEk(oS, k):  # after any alpha has changed update the new value in the cache
    Ek = calcEk(oS, k)
    oS.eCache[k] = [1, Ek]


# 类似于smoSimple()
def innerL(i, oS):
    Ei = calcEk(oS, i)
    if ((oS.labelMat[i] * Ei < -oS.tol) and (oS.alphas[i] < oS.C)) or (
            (oS.labelMat[i] * Ei > oS.tol) and (oS.alphas[i] > 0)):
        j, Ej = selectJ(i, oS, Ei)  # this has been changed from selectJrand
        alphaIold = oS.alphas[i].copy()
        alphaJold = oS.alphas[j].copy()
        if (oS.labelMat[i] != oS.labelMat[j]):
            L = max(0, oS.alphas[j] - oS.alphas[i])
            H = min(oS.C, oS.C + oS.alphas[j] - oS.alphas[i])
        else:
            L = max(0, oS.alphas[j] + oS.alphas[i] - oS.C)
            H = min(oS.C, oS.alphas[j] + oS.alphas[i])
        if L == H:
            # print("L==H")
            return 0
        eta = 2.0 * oS.X[i, :] * oS.X[j, :].T - oS.X[i, :] * oS.X[i, :].T - oS.X[j, :] * oS.X[j,
                                                                                         :].T  # changed for kernel
        if eta >= 0:
            # print("eta>=0")
            return 0
        oS.alphas[j] -= oS.labelMat[j] * (Ei - Ej) / eta
        oS.alphas[j] = clipAlpha(oS.alphas[j], H, L)
        updateEk(oS, j)  # added this for the Ecache
        if (abs(oS.alphas[j] - alphaJold) < 0.00001):
            # print("j not moving enough")
            return 0
        oS.alphas[i] += oS.labelMat[j] * oS.labelMat[i] * (alphaJold - oS.alphas[j])  # update i by the same amount as j
        updateEk(oS, i)  # added this for the Ecache                    #the update is in the oppostie direction
        b1 = oS.b - Ei - oS.labelMat[i] * (oS.alphas[i] - alphaIold) * oS.X[i, :] * oS.X[i, :].T - oS.labelMat[j] * (
                    oS.alphas[j] - alphaJold) * oS.X[i, :] * oS.X[j, :].T
        b2 = oS.b - Ej - oS.labelMat[i] * (oS.alphas[i] - alphaIold) * oS.X[i, :] * oS.X[j, :].T - oS.labelMat[j] * (
                    oS.alphas[j] - alphaJold) * oS.X[j, :] * oS.X[j, :].T

        if (0 < oS.alphas[i]) and (oS.C > oS.alphas[i]):
            oS.b = b1
        elif (0 < oS.alphas[j]) and (oS.C > oS.alphas[j]):
            oS.b = b2
        else:
            oS.b = (b1 + b2) / 2.0
        return 1
    else:
        return 0

def kernelTrans(X, A, kTup): #calc the kernel or transform data to a higher dimensional space
    m,n = shape(X)
    K = mat(zeros((m,1)))
    if kTup[0]=='lin': K = X * A.T   #linear kernel
    elif kTup[0]=='rbf':
        for j in range(m):
            deltaRow = X[j,:] - A
            K[j] = deltaRow*deltaRow.T
        K = exp(K/(-1*kTup[1]**2)) #divide in NumPy is element-wise not matrix like Matlab
    # 如果遇到无法识别的元组就抛出异常
    else: raise NameError('Houston We Have a Problem -- \
    That Kernel is not recognized')
    return K


def smoP(dataMatIn, classLabels, C, toler, maxIter, kTup=('lin', 0)):
    oS = optStruct(mat(dataMatIn), mat(classLabels).transpose(), C, toler) #初始化一个结构类，用于初始化相关变量
    iter = 0 # 迭代次数
    entireSet = True # 是否使用全部数据
    alphaPairsChanged = 0 # alpha
    # 遍历所有的值
    while (iter < maxIter) and ((alphaPairsChanged > 0) or (entireSet)):
        alphaPairsChanged = 0
        if entireSet:
            for i in range(oS.m):
                alphaPairsChanged += innerL(i, oS) # 返回0或者1
            iter += 1
        # 遍历边界值
        else:
            nonBoundIs = nonzero((oS.alphas.A > 0) * (oS.alphas.A < C))[0]
            for i in nonBoundIs:
                alphaPairsChanged += innerL(i, oS)
            iter += 1
        if entireSet:
            entireSet = False
        elif (alphaPairsChanged == 0):
            entireSet = True
    return oS.b, oS.alphas



def loadImages(data,label):
    hwLabels = []
    m = len(data)
    trainingMat = zeros((m,784))
    j=0
    for i in range(m):
        if label[i] == 9:
            hwLabels.append(-1)
            trainingMat[j, :] = np.array(data[i]).reshape(784,)
        elif label[i]==1:
            hwLabels.append(1)
            trainingMat[j, :] = np.array(data[i]).reshape(784, )
        else:
            continue
        j+=1
    return trainingMat[:j], hwLabels

def testDigits(train_data,train_label,test_data,test_label,kTup=('rbf', 10)):
    dataArr,labelArr = loadImages(train_data,train_label) # 获取数据
    b,alphas = smoP(dataArr, labelArr, 200, 0.0001, 100000, kTup) # 获取b和alpha
    datMat=mat(dataArr)
    labelMat = mat(labelArr).transpose()
    svInd=nonzero(alphas.A>0)[0]
    sVs=datMat[svInd]
    labelSV = labelMat[svInd]
    m,n = shape(datMat)
    errorCount = 0
    for i in range(m):
        kernelEval = kernelTrans(sVs,datMat[i,:],kTup) # 计算矩阵K
        predict=kernelEval.T * multiply(labelSV,alphas[svInd]) + b # 获取预测值
        if sign(predict)!=sign(labelArr[i]): errorCount += 1  # 使用sign函数，计数分类错误数
    a=(float(errorCount)/m)
    dataArr,labelArr = loadImages(test_data,test_label)
    errorCount = 0
    datMat=mat(dataArr)
    m,n = shape(datMat)
    for i in range(m):
        kernelEval = kernelTrans(sVs,datMat[i,:],kTup)
        predict=kernelEval.T * multiply(labelSV,alphas[svInd]) + b
        if sign(predict)!=sign(labelArr[i]): errorCount += 1
    b=(float(errorCount)/m)
    return a,b, shape(sVs)[0]


def test2classify():
    train_data,train_label, test_data,test_label = test_SVM()
    print("内核，设置   训练错误率   测试错误率  支持向量数   耗时", end="\t\n")
    b_=6
    for i in [10, 50, 100]:
        begin_=time.time()
        a, b, c = testDigits(train_data,train_label, test_data,test_label,kTup=('rbf', i))
        end_=time.time()
        print("RBF,%.1f"%(i)+    ' '*b_+" %.2f    %.2f    %d     %.2f" % (a, b, c,end_-begin_), end="\t\n")
        b_-=int(i)//50




def loadImages_1(data,label,n):
    hwLabels = []
    m = len(data)
    trainingMat = zeros((m,784))
    j=0
    for i in range(m):
        if label[i] == n:
            hwLabels.append(1)
            trainingMat[i, :] = np.array(data[i]).reshape(784,)
        else:
            hwLabels.append(-1)
            trainingMat[i, :] = np.array(data[i]).reshape(784, )
    return trainingMat, hwLabels

def test_mutiplyclassify():
    train_data,train_label, test_data,test_label = test_SVM()
    print("内核，设置   训练错误率   测试错误率  支持向量数   耗时", end="\t\n")
    b_=6
    for i_ in [10, 50, 100]:
        begin_=time.time()
        kTup=('rbf', i_)
        error_total_train=[]
        error_total_test=[]
        for n_ in range(10):
            dataArr, labelArr = loadImages_1(train_data, train_label,n_)  # 获取数据
            b, alphas = smoP(dataArr, labelArr, 200, 0.0001, 100000, kTup)  # 获取b和alpha
            datMat = mat(dataArr)
            labelMat = mat(labelArr).transpose()
            svInd = nonzero(alphas.A > 0)[0]
            sVs = datMat[svInd]
            labelSV = labelMat[svInd]
            m, n = shape(datMat)
            errorCount = 0
            for i in range(m):
                kernelEval = kernelTrans(sVs, datMat[i, :], kTup)  # 计算矩阵K
                predict = kernelEval.T * multiply(labelSV, alphas[svInd]) + b  # 获取预测值
                if sign(predict) != sign(labelArr[i]): errorCount += 1  # 使用sign函数，计数分类错误数
            error_total_train.append(errorCount)
            # print("%d's train error is %d"%(n_,errorCount))
            dataArr, labelArr = loadImages_1(test_data, test_label,n_)
            errorCount = 0
            datMat = mat(dataArr)
            m, n = shape(datMat)
            for i in range(m):
                kernelEval = kernelTrans(sVs, datMat[i, :], kTup)
                predict = kernelEval.T * multiply(labelSV, alphas[svInd]) + b
                if sign(predict) != sign(labelArr[i]): errorCount += 1
            error_total_test.append(errorCount)
            # print("%d's test error is %d" % (n_, errorCount))
            c=shape(sVs)[0]
        end_ = time.time()
        print("RBF,%.1f" % (i_) + ' ' * b_ + " %.2f    %.2f    %d     %.2f" % (sum(error_total_train)/10000, sum(error_total_test)/100000, c, end_ - begin_), end="\t\n")
        b_ -= int(i_) // 50






if __name__ == '__main__':
    test2classify() # 二分类
    test_mutiplyclassify() # 多分类


