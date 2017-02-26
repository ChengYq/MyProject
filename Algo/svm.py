# coding= utf-8
from sklearn import svm


def originSVM(dataSet):
    # dataset是训练集，还没有区分feature 和 label。
    # 其中，feature还没有进行feature selection,所以输入的dataSet应该先去掉冗余特征
    feature = []
    label = []
    for i in dataSet:
        feature.append(i[:-1])
        label.append(i[-1])

    clf = svm.SVC()
    clf.fit(feature, label)


def test():
    from PreProcess import createDataset
    from FeatureSelection import bagging
    from os import path
    filePath = path.abspath(path.join(path.dirname(__file__), path.pardir, r'DataSet', r'MDP', r'PROMISE', r'cm1.arff'))
    trainset, testset = createDataset.createDataSet(filePath, 5)
    featureSet = bagging.bagIt(trainset)

    # TODO: need to complete
