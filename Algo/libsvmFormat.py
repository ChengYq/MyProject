# coding=utf-8
# 参考：http://www.csie.ntu.edu.tw/~cjlin/papers/guide/data/
# 参考：http://www.cnblogs.com/Finley/p/5329417.html

def formatlib(feature, label):
    # 本函数的目的是：生成对应libsvm可用的数据格式
    # 输入：应当是训练集（已经经过特征选择过的）
    y = []
    x = []
    for i in range(len(feature)):
        if label[i] == u'true':
            a = 1
        else:
            a = -1
        y.append(a)
        x.append(dict(zip(range(1, len(feature[1]) + 1), feature[i])))
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

    features = []
    labels = []
    for i in r:
        # print list(i)[:-1]
        features.append(list(i)[:-1])
        labels.append(list(i)[-1])


    # formatlib(r)
    formatlib(features, labels)


if __name__ == '__main__':
    test()
