# coding=utf-8
from FeatureSelection import bagging
from FeatureSelection import reliefF
from PreProcess import createDataset
from os import path
import numpy as np



def selectedSet(dataSet):
    # 本函数的目的：根据前边的特征选择部分，返回一个特征选择后的数据集
    # 基本思路： bagging + featureSelection
    # 输入 bagging后的数据集子集
    # 输出：进行了特征选择后的数据集（特征子集）
    count = [0 for i in range(len(dataSet[1]) - 1)]
    for i in range(20):
        # range里边就是bagging的次数
        #
        baggedDataSet = bagging.bagIt(dataSet)
        index = reliefF.avgWeight(baggedDataSet, 10)
        res = sorted(zip(index, range(len(index))), key=lambda x: x[0], reverse=True)

        featureNum = []

        for ii in res:
            # print type(ii)
            # print ii[1]
            if ii[0] >= 0:
                featureNum.append(ii[1])
        # print i

        print featureNum
        # print len(baggedDataSet[1])-1

        for i in featureNum:
            count[i] += 1

    print '--------------------'
    # print filter(lambda x:x>=15,count)
    print len(count)
    feature_index = [i for i in range(len(count)) if count[i] >= 12]

    print len(feature_index)

    aaa = np.array(dataSet)
    res = aaa[:, feature_index]

    return res

    # for i in rangepython  二维数组 列索引(len(dataSet)):
    #     if i in feature_index:


def test():
    filePath = path.abspath(path.join(path.dirname(__file__), path.pardir, r'DataSet', r'MDP', r'PROMISE', r'cm1.arff'))
    trainset, testset = createDataset.createDataSet(filePath, 5)
    selectedSet(trainset)


if __name__ == '__main__':
    test()
