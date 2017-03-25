# coding=utf-8
import numpy as np
from scipy.stats import pearsonr

def corr(feature):
    toDelete = []
    row_num, col_num = feature.shape
    correlation = np.zeros((col_num, col_num))
    pvalue = np.zeros((col_num, col_num))
    for i in range(col_num):
        for j in range(col_num):
            c, p = pearsonr(feature[:, i], feature[:, j])
            correlation[i, j] = c
            pvalue[i, j] = p

    for i in range(row_num):
        for j in range(i + 1, col_num):
            if (i not in toDelete) and (j not in toDelete):
                if (correlation[i, j] > 0.95) and (pvalue[i, j] < 0.01):
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
    # trainset, x_min, x_max = minmaxscaler(train_feature)
    # testset = minmaxscaler(test_feature, x_feature_min=x_min, x_feature_max=x_max)

    from sklearn import preprocessing
    trainset = preprocessing.scale(train_feature)
    testset = preprocessing.scale(test_feature)

    toDelete = corr(trainset)

    print toDelete


#####  old
# toDelete = []
# corr_matrix = np.corrcoef(feature, rowvar=0)
# row, col = corr_matrix.shape
# for i in range(row):
#     for j in range(i + 1, col):
#         if (i not in toDelete) and (j not in toDelete):
#             if corr_matrix[i, j] > 0.95:
#                 toDelete.append(j)
# return toDelete

if __name__ == '__main__':
    test()
