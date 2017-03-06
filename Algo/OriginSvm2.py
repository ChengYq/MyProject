# coding= utf-8
# 参考： http://www.techv5.com/topic/289/
# 参考： http://www.csie.ntu.edu.tw/~cjlin/libsvm/faq.html#/Q04:_Training_and_prediction

# 2 这个版本是为了生成arff文件的！
# 我的目的是： 自己bagging后的数据集生成arff, 然后把它放进easy.py进行训练。先看看我的方法有没有进步。
# 我自己的方法：参看自己的说明文档：    /home/chyq/Document/MyProject/DataSet/MDP

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
    from FeatureSelection import afterFeatureSelection2
    from PreProcess.minmax2 import minmaxscaler
    from PreProcess.createDataset import featureAndLabel

    filePath = path.abspath(path.join(path.dirname(__file__), path.pardir, r'DataSet', r'MDP', r'D1', r'KC1.arff'))

    data, trainset, testset, relation, attribute = createDataSet(filePath, 5)

    # featureSet = bagging.bagIt(trainset)

    train_feature, train_label = featureAndLabel(trainset)
    test_feature, test_label = featureAndLabel(testset)

    trainset, x_min, x_max = minmaxscaler(train_feature)

    testset = minmaxscaler(test_feature, x_feature_min=x_min, x_feature_max=x_max)

    featured_trainset, attribute = afterFeatureSelection2.selectedSet(trainset, train_label, attribute, train_feature)

    arff_obj = {'relation': relation, 'attributes': attribute, 'data': featured_trainset}

    import arff

    to = arff.dumps(arff_obj)
    try:
        f = open('/home/chyq/Document/MyProject/DataSet/MDP/my/my_kc1.arff', 'w')
        f.write(to)
    finally:
        f.close()





        # 这里生成一下arff文件，


        # 到此已经执行成功！！！


if __name__ == '__main__':
    test()
