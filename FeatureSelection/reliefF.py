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
    # 本函数的内容：就是实现reliefF的细节，不必追究
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
        pass


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
        # weightSeq = weight / (iter_ratio * n_samples)

        # 以下代码，是用来对特征的分值进行排序的，最终的结果存储在index 里，
        # index从左到右，存储了特征重要性从高到低。
        # weightDict = zip(weightSeq, range(len(weightSeq)))
        # res = sorted(weightDict, key=lambda x: x[0], reverse=True)
        #
        # index = []
        # for i in res:
        #     index.append(i[1])
    #
    # print index
    # return index
    # print weight / (iter_ratio * n_samples)
    return weight / (iter_ratio * n_samples)


def avgWeight(featureSet, ite_time):
    # ite_time是计算的次数,fit函数里边的ratio是采样的比例，目前是1
    # 之所以要多次求平均，并不是因为采样比例的问题，
    # 我们可以发现，即便ratio选择为1，结果还是会不一样，因为这个算法有一个随机选择中心的过程
    # 返回值：就是每个权值的打分
    # 输入：bagging后的数据集，计算次数
    # 返回：ite_time次计算权值的平均值

    allRes = []
    for i in range(ite_time):
        weight = fit(featureSet, 1)
        allRes.append(weight)
    aaa = np.array(allRes)

    return np.mean(aaa, axis=0)  # 这里需要求平均值！！



def test():
    from PreProcess import createDataset
    from FeatureSelection import bagging
    from os import path
    filePath = path.abspath(path.join(path.dirname(__file__), path.pardir, r'DataSet', r'MDP', r'PROMISE', r'cm1.arff'))
    trainset, testset = createDataset.createDataSet(filePath, 5)
    featureSet = bagging.bagIt(trainset)

    avgWeight(featureSet, 20)

if __name__ == '__main__':
    test()
