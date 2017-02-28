# coding= utf-8
# 参考： http://www.techv5.com/topic/289/


from Algo.libsvm.myPython.svm import *
from Algo.libsvm.myPython.svmutil import *
# import Algo.libsvm.myPython.svm
# import Algo.libsvm.myPython.svmutil
from Algo import libsvmFormat


def originSVM(trainSet, testSet):
    # dataset是训练集，还没有区分feature 和 label。
    # 其中，feature还没有进行feature selection,所以输入的dataSet应该先去掉冗余特征
    # feature = []
    # label = []
    # for i in dataSet:
    #     feature.append(i[:-1])
    #     label.append(i[-1])
    #
    # clf = svm.SVC()
    # clf.fit(feature, label)

    x, y = libsvmFormat.formatlib(trainSet)
    xtest, ytest = libsvmFormat.formatlib(testSet)
    prob = svm_problem(y, x)
    param = svm_parameter('-t 0 -c 4 -b 1')
    model = svm_train(prob, param)
    p_label, p_acc, p_val = svm_predict(ytest, xtest, model)
    print p_label, p_acc, p_val





def test():
    from PreProcess import createDataset
    from FeatureSelection import bagging
    from os import path
    from FeatureSelection import afterFeatureSelection
    filePath = path.abspath(path.join(path.dirname(__file__), path.pardir, r'DataSet', r'MDP', r'PROMISE', r'cm1.arff'))
    trainset, testset = createDataset.createDataSet(filePath, 5)
    # featureSet = bagging.bagIt(trainset)
    featured_trainset = afterFeatureSelection.selectedSet(trainset)
    originSVM(featured_trainset, testset)


if __name__ == '__main__':
    test()
