# coding=utf-8
# http://blog.csdn.net/kryolith/article/details/40849483
import numpy as np
from random import randrange

from sklearn.datasets import make_classification
from sklearn.preprocessing import normalize


def distanceNorm(Norm, D_value):
    # 这里的Nrom代表了范数的类型，Dvalue在后边的函数中有所用

    # Norm for distance
    if Norm == '1':
        counter = np.absolute(D_value)
        counter = np.sum(counter)
    elif Norm == '2':
        counter = np.power(D_value, 2)
        counter = np.sum(counter)
        counter = np.sqrt(counter)
    elif Norm == 'inf':
        counter = np.absolute(D_value)
        counter = np.max(counter)
    else:
        raise Exception('只能1,2,inf')

    return counter


def fit(dataSet, iter_ratio):
    # 首先，讲feature 和 label 分离出来
    features = []
    labels = []
    for i in dataSet:
        # print list(i)[:-1]
        features.append(list(i)[:-1])
        labels.append(list(i)[-1])

    # 归一化
    features = normalize(X=features, norm='l2', axis=0)

    # 以下代码用来产生reliefF函数
    (n_samples, n_features) = np.shape(features)
    distance = np.zeros((n_samples, n_samples))
    weight = np.zeros(n_features)

    if iter_ratio >= 0.5:
        # compute distance
        for index_i in xrange(n_samples):
            for index_j in xrange(index_i + 1, n_samples):
                D_value = features[index_i] - features[index_j]
                distance[index_i, index_j] = distanceNorm('2', D_value)
        distance += distance.T
    else:
        pass;


        # start iteration
    for iter_num in xrange(int(iter_ratio * n_samples)):
        # print iter_num;
        # initialization
        nearHit = []
        nearMiss = []
        distance_sort = list()

        # random extract a sample
        index_i = randrange(0, n_samples, 1)
        self_features = features[index_i]

        # search for nearHit and nearMiss
        if iter_ratio >= 0.5:
            distance[index_i, index_i] = np.max(distance[index_i])  # filter self-distance
            for index in xrange(n_samples):
                distance_sort.append([distance[index_i, index], index, labels[index]])
        else:
            # compute distance respectively
            distance = np.zeros(n_samples)
            for index_j in xrange(n_samples):
                D_value = features[index_i] - features[index_j]
                distance[index_j] = distanceNorm('2', D_value)
            distance[index_i] = np.max(distance)  # filter self-distance
            for index in xrange(n_samples):
                distance_sort.append([distance[index], index, labels[index]])

        distance_sort.sort(key=lambda x: x[0])
        for index in xrange(n_samples):
            if nearHit == [] and distance_sort[index][2] == labels[index_i]:
                # nearHit = distance_sort[index][1];
                nearHit = features[distance_sort[index][1]]
            elif nearMiss == [] and distance_sort[index][2] != labels[index_i]:
                # nearMiss = distance_sort[index][1]
                nearMiss = features[distance_sort[index][1]]
            elif nearHit != [] and nearMiss != []:
                break
            else:
                continue
                # update weight
        weight = weight - np.power(self_features - nearHit, 2) + np.power(self_features - nearMiss, 2)

        # weightedSeq存储了特征的重要性的打分值
        weightSeq = weight / (iter_ratio * n_samples)

        # 以下代码，是用来对特征的分值进行排序的，最终的结果存储在index 里，
        # index从左到右，存储了特征重要性从高到低。
        weightDict = zip(weightSeq, range(len(weightSeq)))
        res = sorted(weightDict, key=lambda x: x[0], reverse=True)

        index = []
        for i in res:
            index.append(i[1])

    print index
    return index
    # print weight / (iter_ratio * n_samples)
    # return weight / (iter_ratio * n_samples)


def test():
    from PreProcess import createDataset
    from FeatureSelection import bagging
    from os import path
    filePath = path.abspath(path.join(path.dirname(__file__), path.pardir, r'DataSet', r'MDP', r'PROMISE', r'cm1.arff'))
    trainset, testset = createDataset.createDataSet(filePath, 5)
    featureSet = bagging.bagIt(trainset)

    for x in xrange(1, 20):
        # 这个地方非常重要！因为relefF算法需要多次执行，看以下平均值
        weight = fit(featureSet, 0.6)
        # 这个0.6，就是按照比例抽样的意思，如果是1，那么就全部抽样了


if __name__ == '__main__':
    test()
