# coding=utf-8
# 这个代码是为了 Bagging + Correlation 进行使用的
import numpy as np
from FeatureSelection import bagging
from FeatureSelection import reliefF
from FeatureSelection import correlation


def selectedSet(feature, label, attribute, origin_faeature):
    # 本函数的目的：根据前边的特征选择部分，返回一个特征选择后的数据集
    # 基本思路： bagging + featureSelection
    # 输入 测试集！不要输入bagging后的！！！
    # 输出：进行了特征选择后的数据集（特征子集）

    # count有点哈希的意味，初始全0，长度等于feature的个数。
    count = [0 for i in range(len(feature[1]))]
    # range里边就是bagging的次数
    for i in range(20):
        # 进行bagging操作，将bagging后的结果返回给baggedDataSet
        baggedDataSet = bagging.bagIt(feature, label)
        # 计算reliefF，进行10次，然后求权值的平均值，
        # index = reliefF.avgWeight(baggedDataSet, 10)
        # 将权值进行排序，zip方法是为了返回下标方便些
        baggedDataSet = np.array(baggedDataSet)
        bagged_features = baggedDataSet[:, :-1]
        toDelete = correlation.corr(bagged_features)

        feature_index = range(len(feature[1]))
        for i in feature_index:
            if i in toDelete:
                feature_index.remove(i)

        print feature_index

        # 将fearueNum里边的数遍历，将对应下标+=1
        for i in feature_index:
            count[i] += 1

    print '--------------------'
    # print filter(lambda x:x>=15,count)
    print len(count)

    # 列表生成器。 将那些出现次数大于12的下标拿出来，存储进feature_index中
    feature_index = [i for i in range(len(count)) if count[i] >= 0]

    print feature_index
    print len(feature_index)

    # feature_index.append(-1)  # 这个代码是为了增加-1这个索引，就是将label也要存进去。

    final_attribute = []

    for i in range(len(attribute)):
        if i in feature_index:
            final_attribute.append(attribute[i])

    final_attribute.append((u'Defective', [u'1.0', u'-1.0']))

    aaa = np.c_[origin_faeature, label]
    # 为了生成特征子集，转为ndarray就可以使用切片功能了，相当的方便
    feature_index.append(-1)

    resSet = aaa[:, feature_index]
    resSet = resSet.tolist()
    print resSet
    # 这里的res是List类型的
    return resSet, final_attribute

    # for i in rangepython  二维数组 列索引(len(dataSet)):
    #     if i in feature_index:


def test():
    from PreProcess.createDataset import createDataSet
    from os import path
    from FeatureSelection import afterFeatureSelection3
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

    featured_trainset, featured_attribute = afterFeatureSelection3.selectedSet(trainset, train_label, attribute,
                                                                               train_feature)


if __name__ == '__main__':
    test()
