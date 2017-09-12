# coding=utf-8
# kNN weight
from numpy import *
import operator


def create_weightknn(featured_trainset, train_label, k):
    # 输入 train_feature, train_label 都是numpy类型的
    # 输出：noise_weight 一个list，反映了权值
    weight = []
    for i in range(len(featured_trainset)):
        temp = classify(featured_trainset[i], featured_trainset, train_label, k)
        for ii in temp:
            if ii[0] == train_label[i]:
                if ii[1] >= k / 2.0:
                    a = 1
                elif ii[0] <= k / 10.0:
                    a = 0
                else:
                    a = 0.01 + (1 - 0.01) * (ii[0] - k / 10.0) / (k / 2.0 - k / 10.0)
                weight.append(a)
    return weight


def classify(inX, dataSet, labels, k):
    inX = inX.tolist()
    labels = labels.tolist()
    dataSetSize = dataSet.shape[0]
    diffMat = tile(inX, (dataSetSize, 1)) - dataSet
    sqDiffMat = diffMat ** 2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances ** 0.5
    sortedDistIndicies = distances.argsort()
    classCount = {}
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
    sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount
