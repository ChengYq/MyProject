# coding=utf-8
# 本模块用来读取要研究的arff文件以及属性信息
# 函数readArff的作用：读取arff文件，输入：文件的路径。输出：data:数据内容，mata: 属性信息

import random
from scipy.io import arff


def readArff(path):
    # data 就是要读取的文件
    # meta 就是关于该文件的一些信息
    data, meta = arff.loadarff(path)
    return data, meta


def createDataSet(arffPath, fold):
    trainSet = []
    testSet = []
    data, meta = arff.loadarff(arffPath)
    seq = range(len(data))
    random.shuffle(seq)
    testNum = len(data) / fold
    # print len(data)-testNum
    # print len(seq[:testNum])
    # print len(seq[testNum:])

    for i in seq[:testNum]:
        testSet.append(data[i])
    for j in seq[testNum:]:
        trainSet.append(data[j])

    return trainSet, testSet


if __name__ == '__main__':
    from os import path

    # # print os.path.dirna
    filePath = path.abspath(path.join(path.dirname(__file__), path.pardir, r'DataSet', r'MDP', r'PROMISE', r'cm1.arff'))
    # data,meta=readArff(filePath)
    # print len(data)
    train, test = createDataSet(filePath, 5)
    print train
    print test
