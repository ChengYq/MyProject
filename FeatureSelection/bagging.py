# coding=utf-8
import numpy as np
def bagIt(feature, label):
    # 输入：dataSet为ndarray类型的,输入应该是训练集
    # 返回：经过bagging后的数据集子集,含有label
    noDefectCount = 0  # 初始化无缺陷数据个数
    defectCount = 0  # 初始化有缺陷数据个数
    defectSet = []  # 初始化有缺陷的数据 ，最后用于记录有缺陷和无缺陷的数据
    noDefectSet = []  # 初始化无缺陷的数据


    for i in range(len(label)):
        #分别记录有、无缺陷
        if label[i] == 1:
            defectCount += 1
            defectSet.append(np.append(feature[i], label[i]))

        else:
            noDefectCount += 1
            noDefectSet.append(np.append(feature[i], label[i]))

    # 产生随机数
    seq = range(len(noDefectSet))
    from random import shuffle
    shuffle(seq)
    # print seq[:defectCount], len(seq[:defectCount])

    # print len(seq)

    for i in seq[:defectCount]:
        defectSet.append(noDefectSet[i])
    # 请注意，这里的defectSet 追加了同样个数的无缺陷的数据

    featureSelectionSet = defectSet

    # print defectSet
    return featureSelectionSet


def test():
    from PreProcess import createDataset
    from os import path

    filePath = path.abspath(path.join(path.dirname(__file__), path.pardir, r'DataSet', r'MDP', r'PROMISE', r'cm1.arff'))
    train, test = createDataset.createDataSet(filePath, 5)
    f = bagIt(train)
    print f

if __name__ == '__main__':
    test()
