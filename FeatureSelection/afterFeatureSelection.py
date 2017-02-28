# coding=utf-8

import numpy as np
from FeatureSelection import bagging
from FeatureSelection import reliefF

def selectedSet(dataSet):
    # 本函数的目的：根据前边的特征选择部分，返回一个特征选择后的数据集
    # 基本思路： bagging + featureSelection
    # 输入 测试集！不要输入bagging后的！！！
    # 输出：进行了特征选择后的数据集（特征子集）

    # count有点哈希的意味，初始全0，长度等于feature的个数。
    count = [0 for i in range(len(dataSet[1]) - 1)]
    # range里边就是bagging的次数
    for i in range(20):
        # 进行bagging操作，将bagging后的结果返回给baggedDataSet
        baggedDataSet = bagging.bagIt(dataSet)
        # 计算reliefF，进行10次，然后求权值的平均值，
        index = reliefF.avgWeight(baggedDataSet, 10)
        # 将权值进行排序，zip方法是为了返回下标方便些
        res = sorted(zip(index, range(len(index))), key=lambda x: x[0], reverse=True)
        # featureNum记录了那些被特征选择选中的下标号码
        featureNum = []

        # for循环是为了选择出权值大于1的那些特征，将其记录在featureNum中
        for ii in res:
            # print type(ii)
            # print ii[1]
            if ii[0] >= 0:
                featureNum.append(ii[1])
        # print i

        print featureNum
        # print len(baggedDataSet[1])-1

        # 将fearueNum里边的数遍历，将对应下标+=1
        for i in featureNum:
            count[i] += 1

    print '--------------------'
    # print filter(lambda x:x>=15,count)
    print len(count)

    # 列表生成器。 将那些出现次数大于12的下标拿出来，存储进feature_index中
    feature_index = [i for i in range(len(count)) if count[i] >= 12]

    print len(feature_index)

    feature_index.append(-1)  # 这个代码是为了增加-1这个索引，就是将label也要存进去。

    # 为了生成特征子集，转为ndarray就可以使用切片功能了，相当的方便
    aaa = np.array(dataSet)
    resSet = aaa[:, feature_index]
    resSet = resSet.tolist()
    print resSet
    # 这里的res是List类型的
    return resSet

    # for i in rangepython  二维数组 列索引(len(dataSet)):
    #     if i in feature_index:


def test():
    from PreProcess import createDataset
    from os import path
    filePath = path.abspath(path.join(path.dirname(__file__), path.pardir, r'DataSet', r'MDP', r'PROMISE', r'cm1.arff'))
    trainset, testset = createDataset.createDataSet(filePath, 5)
    r = selectedSet(trainset)


if __name__ == '__main__':
    test()
