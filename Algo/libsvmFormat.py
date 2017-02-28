# coding=utf-8
# 参考：http://www.csie.ntu.edu.tw/~cjlin/papers/guide/data/
# 参考：http://www.cnblogs.com/Finley/p/5329417.html

def formatlib(dataSet):
    # 本函数的目的是：生成对应libsvm可用的数据格式
    # 输入：应当是训练集（已经经过特征选择过的）
    y = []
    x = []
    for i in dataSet:
        if i[-1] == u'true':
            a = 1
        else:
            a = -1
        y.append(a)
        x.append(dict(zip(range(1, len(dataSet[1])), map(float, i[:-1]))))
    print y
    print x
    return x, y


def test():
    from PreProcess import createDataset
    from os import path
    from FeatureSelection import afterFeatureSelection
    filePath = path.abspath(path.join(path.dirname(__file__), path.pardir, r'DataSet', r'MDP', r'PROMISE', r'cm1.arff'))
    trainset, testset = createDataset.createDataSet(filePath, 5)
    r = afterFeatureSelection.selectedSet(trainset)

    # formatlib(r)
    formatlib(testset)


if __name__ == '__main__':
    test()
