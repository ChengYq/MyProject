# coding= utf-8
# 参考： http://www.techv5.com/topic/289/
# 参考： http://www.csie.ntu.edu.tw/~cjlin/libsvm/faq.html#/Q04:_Training_and_prediction


from Algo.libsvm.myPython.svm import *
from Algo.libsvm.myPython.svmutil import *
from Algo import libsvmFormat
from sklearn.preprocessing import normalize


def originSVM(trainSet, testSet):
    # dataset是训练集，还没有区分feature 和 label。
    # 其中，feature还没有进行feature selection,所以输入的dataSet应该先去掉冗余特征

    train_features = []
    train_labels = []
    for i in trainSet:
        # print list(i)[:-1]
        train_features.append(list(i)[:-1])
        train_labels.append(list(i)[-1])

    train_features = normalize(X=train_features, norm='l2', axis=0).tolist()

    test_features = []
    test_labels = []
    for i in testSet:
        # print list(i)[:-1]
        test_features.append(list(i)[:-1])
        test_labels.append(list(i)[-1])
    test_features = normalize(X=test_features, norm='l2', axis=0).tolist()

    x, y = libsvmFormat.formatlib(train_features, train_labels)
    xtest, ytest = libsvmFormat.formatlib(test_features, test_labels)
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

    print "--------------------------------------------"

    originSVM(trainset, testset)



if __name__ == '__main__':
    test()
