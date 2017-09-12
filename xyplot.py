# coding=utf-8

import numpy as np
import matplotlib.pyplot as plt


def test():
    from PreProcess.createDataset import createDataSet
    from os import path
    from FeatureSelection import afterFeatureSelection2
    from PreProcess.minmax2 import minmaxscaler
    from PreProcess.createDataset import featureAndLabel

    # filePath = path.abspath(path.join(path.dirname(__file__), path.pardir, r'DataSet', r'MDP', r'D1', r'KC1.arff'))

    filePath = r'/home/chyq/Document/MyProject/DataSet/MDP/D1/KC1.arff'
    data, trainsetWithLabel, testsetWithLabel, relation, attribute = createDataSet(filePath, 5)

    # featureSet = bagging.bagIt(trainset)

    # 分离出训练集、测试集的feature, label
    train_feature, train_label = featureAndLabel(trainsetWithLabel)
    test_feature, test_label = featureAndLabel(testsetWithLabel)

    # 做normalization 得出的结果在trainset, test。
    trainset, x_min, x_max = minmaxscaler(train_feature, lower=-1)
    testset = minmaxscaler(test_feature, x_feature_min=x_min, x_feature_max=x_max)

    from sklearn import preprocessing
    # trainset = preprocessing.scale(train_feature)
    # testset = preprocessing.scale(test_feature)


    # x = list(trainset[:, 12])
    # y = list(trainset[:, 16])
    # plt.scatter(x, y)
    plt.hist(trainset[:, 13])
    plt.show()


if __name__ == '__main__':
    test()
