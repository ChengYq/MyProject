# coding=utf-8

from scipy.stats import spearmanr
import numpy as np


def spearman(feature):
    toDelete = []
    row_num, col_num = feature.shape
    spear = np.zeros((col_num, col_num))
    pvalue = np.zeros((col_num, col_num))
    for i in range(col_num):
        for j in range(col_num):
            c, p = spearmanr(feature[:, i], feature[:, j])
            spear[i, j] = c
            pvalue[i, j] = p

    for i in range(row_num):
        for j in range(i + 1, col_num):
            if (i not in toDelete) and (j not in toDelete):
                if (spear[i, j] > 0.95) and (pvalue[i, j] < 0.01):
                    toDelete.append(j)

    return toDelete


def test():
    from PreProcess.createDataset import createDataSet
    from os import path
    from FeatureSelection import afterFeatureSelection2
    from PreProcess.minmax2 import minmaxscaler
    from PreProcess.createDataset import featureAndLabel

    filePath = path.abspath(path.join(path.dirname(__file__), path.pardir, r'DataSet', r'MDP', r'D1', r'KC1.arff'))

    data, trainsetWithLabel, testsetWithLabel, relation, attribute = createDataSet(filePath, 5)

    # featureSet = bagging.bagIt(trainset)

    # 分离出训练集、测试集的feature, label
    train_feature, train_label = featureAndLabel(trainsetWithLabel)
    test_feature, test_label = featureAndLabel(testsetWithLabel)

    # 做normalization 得出的结果在trainset, test。
    trainset, x_min, x_max = minmaxscaler(train_feature)
    testset = minmaxscaler(test_feature, x_feature_min=x_min, x_feature_max=x_max)

    toDelete = spearman(trainset)

    print toDelete


if __name__ == '__main__':
    test()
