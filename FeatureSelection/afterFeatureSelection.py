# coding=utf-8
from FeatureSelection import bagging
from FeatureSelection import reliefF


def selectedSet(dataSet):
    # 本函数的目的：根据前边的特征选择部分，返回一个特征选择后的数据集
    # 基本思路： bagging + featureSelection
    for i in range(10):
        baggedDataSet = bagging(dataSet)
        for ii in range(10):
            index = reliefF(baggedDataSet)
