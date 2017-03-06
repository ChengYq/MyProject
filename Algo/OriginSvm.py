# coding= utf-8
# 参考： http://www.techv5.com/topic/289/
# 参考： http://www.csie.ntu.edu.tw/~cjlin/libsvm/faq.html#/Q04:_Training_and_prediction

from Algo.libsvm.myPython.svm import *
from Algo.libsvm.myPython.svmutil import *
from Algo import libsvmFormat


def originSVM(train_features, train_labels, test_features, test_labels):
    # dataset是训练集，还没有区分feature 和 label。
    # 其中，feature还没有进行feature selection,所以输入的dataSet应该先去掉冗余特征

    # 以下代码是用来将label和 feature 分开用的
    # train_features = []
    # train_labels = []
    # for i in trainSet:
    #     # print list(i)[:-1]
    #     train_features.append(map(float,list(i)[:-1]))
    #     train_labels.append(list(i)[-1])
    #
    #
    # test_features = []
    # test_labels = []
    # for i in testSet:
    #     # print list(i)[:-1]
    #     test_features.append(list(i)[:-1])
    #     test_labels.append(list(i)[-1])

    x, y = libsvmFormat.formatlib(train_features, train_labels)
    xtest, ytest = libsvmFormat.formatlib(test_features, test_labels)
    prob = svm_problem(y, x)
    param = svm_parameter('-c 32 -g 0.125')  # 这里值得继续研究研究！
    #  问题就是在这里！！ 一般C比较大，gamma取0.5的时候，效果还是可以的
    # D1 的KC1 ,C = 32 gamma=0.125 为最优配置！
    # 实际上，grid里边的C ，就是按照2^N 的方式进行的

    model = svm_train(prob, param)
    predicted_label, predicted_acc, predicted_val = svm_predict(ytest, xtest, model)
    print predicted_label
    return predicted_label


def test():
    from PreProcess.createDataset import createDataSet
    from os import path
    from FeatureSelection import afterFeatureSelection
    from PreProcess.minmax2 import minmaxscaler
    from PreProcess.createDataset import featureAndLabel

    filePath = path.abspath(path.join(path.dirname(__file__), path.pardir, r'DataSet', r'MDP', r'D1', r'KC1.arff'))

    data, trainset, testset, relation, attribute = createDataSet(filePath, 5)

    # featureSet = bagging.bagIt(trainset)

    train_feature, train_label = featureAndLabel(trainset)
    test_feature, test_label = featureAndLabel(testset)

    trainset, x_min, x_max = minmaxscaler(train_feature)

    testset = minmaxscaler(test_feature, x_feature_min=x_min, x_feature_max=x_max)

    featured_trainset, attribute = afterFeatureSelection.selectedSet(trainset, train_label, attribute)

    from Evaluation.eva import evaluationRes

    predicted_label_f = originSVM(featured_trainset, train_label, testset, test_label)

    # print predicted_label_f
    # print testset

    a, b = evaluationRes(predicted_label_f, test_label)

    print a, b

    print "--------------------------------------------"

    predicted_label_nf = originSVM(trainset, train_label, testset, test_label)
    c, d = evaluationRes(predicted_label_nf, test_label)
    print c, d
    # confusion_matrix(testset[-1], predicted_label_nf)


if __name__ == '__main__':
    test()
