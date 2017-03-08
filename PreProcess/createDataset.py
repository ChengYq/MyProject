# coding=utf-8
# 本模块用来读取要研究的arff文件以及属性信息,并且根据交叉验证的方法，产生训练集和测试集

from random import shuffle
from arff import load
import numpy as np


def createDataSet(arffPath, fold):
    # 输入： arffpath： arff文件路径。 fold： 交叉验证中的fold个数
    # 输出： 训练集、测试集
    trainSet = []
    testSet = []
    res = load(open(arffPath, 'rb'))
    data = res['data']
    relation = res['relation']
    attribute = res['attributes']
    print len(data)
    seq = range(len(data))  # 用于随机查找
    shuffle(seq)  # shuffle函数：重新对list进行随机排列
    testNum = len(data) / fold  # testNUM: 测试集的元素个数
    # print len(data)-testNum
    # print len(seq[:testNum])
    # print len(seq[testNum:])

    for i in seq[:testNum]:
        testSet.append(data[i])
    for j in seq[testNum:]:
        trainSet.append(data[j])

    return data, trainSet, testSet, relation, attribute
    # 这个里边的数据，是numpy.float64类型的。需要注意！


def convert(x):
    if (x.lower() == 'true') or (x.lower() == 'y'):
        return 1.0
    elif (x.lower() == 'false') or (x.lower() == 'n'):
        return -1.0
    else:
        raise Exception('label format is wrong')


def featureAndLabel(dataSet):
    # 这个函数是用作label和feature分离的
    # 输入： 含有feature和label的数据集
    # 返回：feature 和 lebel ，都是ndarray类型的,
    # 特别是，label的true 是 +1 ,false 是-1

    features = []
    labels = []
    for i in dataSet:
        # print list(i)[:-1]
        features.append(map(float, list(i)[:-1]))
        labels.append(list(i)[-1])

    labels = map(convert, labels)
    return np.array(features), np.array(labels)


def test():
    from os import path
    # # print os.path.dirna
    filePath = path.abspath(path.join(path.dirname(__file__), path.pardir, r'DataSet', r'MDP', r'PROMISE', r'cm1.arff'))
    # data,meta=readArff(filePath)
    # print len(data)
    trainset, testset = createDataSet(filePath, 5)

    train_feature, train_label = featureAndLabel(trainset)

    test_feature, test_label = featureAndLabel(testset)



if __name__ == '__main__':
    test()
