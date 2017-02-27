# coding=utf-8
def bagIt(dataSet):
    # 输入：dataSet为List类型的,输入应该是训练集
    # 返回：经过bagging后的数据集子集
    noDefectCount = 0  # 初始化无缺陷数据个数
    defectCount = 0  # 初始化有缺陷数据个数
    defectSet = []  # 初始化有缺陷的数据 ，最后用于记录有缺陷和无缺陷的数据
    noDefectSet = []  # 初始化无缺陷的数据
    for i in dataSet:
        #分别记录有、无缺陷
        if i[-1].lower() in ['true', 'y', 'yes']:
            defectCount += 1
            defectSet.append(i)

        elif i[-1].lower() in ['false', 'n', 'no']:
            noDefectCount += 1
            noDefectSet.append(i)

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
