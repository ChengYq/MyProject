# coding=utf-8
# 本模块用来读取要研究的arff文件以及属性信息,并且根据交叉验证的方法，产生训练集和测试集

import random
import arff


def createDataSet(arffPath, fold):
    # 输入： arffpath： arff文件路径。 fold： 交叉验证中的fold个数
    # 输出： 训练集、测试集
    trainSet = []
    testSet = []
    res = arff.load(open(arffPath, 'rb'))
    data = res['data']
    print len(data)
    seq = range(len(data))  # 用于随机查找
    random.shuffle(seq)  # shuffle函数：重新对list进行随机排列
    testNum = len(data) / fold  # testNUM: 测试集的元素个数
    # print len(data)-testNum
    # print len(seq[:testNum])
    # print len(seq[testNum:])

    for i in seq[:testNum]:
        testSet.append(data[i])
    for j in seq[testNum:]:
        trainSet.append(data[j])

    return trainSet, testSet
    # 这个里边的数据，是numpy.float64类型的。需要注意！


def test():
    from os import path
    # # print os.path.dirna
    filePath = path.abspath(path.join(path.dirname(__file__), path.pardir, r'DataSet', r'MDP', r'PROMISE', r'cm1.arff'))
    # data,meta=readArff(filePath)
    # print len(data)
    trainset, testset = createDataSet(filePath, 5)
    print trainset
    print testset


if __name__ == '__main__':
    test()
