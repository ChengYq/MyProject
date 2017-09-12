# coding=utf-8
from WeightGeneral import genetic
from WeightFeature import geneticFeature
from PreProcess.createDataset import createDataSet
from os import path
from FeatureSelection import FeatureSelectionProcess
from PreProcess.minmax2 import minmaxscaler
from PreProcess.createDataset import featureAndLabel
from Fsvmcil import create_weight
from Fsvmcilknn import create_weightknn
import arff
import numpy as np
from Algo.libsvmWeight.python.svmutil import *
import WeightFeature as wf


def baggingAlgo():
    filePath = path.abspath(path.join(path.dirname(__file__), path.pardir, r'DataSet', r'MDP', r'D2', r'PC5.arff'))

    data, trainsetWithLabel, testsetWithLabel, relation, attribute = createDataSet(filePath, 10)

    # 分离出训练集、测试集的feature, label
    train_feature, train_label = featureAndLabel(trainsetWithLabel)
    test_feature, test_label = featureAndLabel(testsetWithLabel)

    # 做normalization 得出的结果在trainset, test。
    trainset, x_min, x_max = minmaxscaler(train_feature)
    testset = minmaxscaler(test_feature, x_feature_min=x_min, x_feature_max=x_max)

    featured_trainset, featured_attribute = FeatureSelectionProcess.selectedSet(trainset, train_label, attribute,
                                                                                trainset)
    featured_trainset = np.array(featured_trainset)
    featured_trainset = featured_trainset[:, :-1]

    import LibsvmFormat
    x, y = LibsvmFormat.formatlib(featured_trainset, train_label)
    xtest, ytest = LibsvmFormat.formatlib(testset, test_label)

    # W = []
    W_Knn = create_weightknn(featured_trainset, train_label, 10)

    best_para_knn = genetic(x, y, xtest, ytest, W_Knn)

    W_fsvmcil = create_weight(featured_trainset, train_label)
    best_para_fsvmcvil = genetic(x, y, xtest, ytest, W_fsvmcil)

    best_para_feature = geneticFeature(x, y, xtest, ytest)

    print best_para_knn, best_para_fsvmcvil, best_para_feature

    #############
    prob = svm_problem(W_Knn, y, x)
    p = '-c {0} -g {1}'.format(best_para_knn[1][0], best_para_knn[1][1])
    para = svm_parameter(p)
    model1 = svm_train(prob, para)
    p_label1, p_acc, p_val = svm_predict(ytest, xtest, model1)

    prob = svm_problem(W_fsvmcil, y, x)
    p = '-c {0} -g {1}'.format(best_para_fsvmcvil[1][0], best_para_fsvmcvil[1][1])
    para = svm_parameter(p)
    model2 = svm_train(prob, para)
    p_label2, p_acc, p_val = svm_predict(ytest, xtest, model2)

    W_featureW = wf.create_weight(featured_trainset)
    prob = svm_problem(W_featureW, y, x)
    p = '-c {0} -g {1}'.format(best_para_feature[1][0], best_para_feature[1][1])
    para = svm_parameter(p)
    model3 = svm_train(prob, para)
    p_label3, p_acc, p_val = svm_predict(ytest, xtest, model3)

    result = []
    for i in range(len(p_label1)):
        judge = 0
        if p_label1[i] == 1.0:
            judge += 1
        if p_label2[i] == 1.0:
            judge += 1
        if p_label3[i] == 1.0:
            judge += 1
        if judge >= 2:
            result.append(1.0)
        else:
            result.append(-1.0)
    return result


if __name__ == "__main__":
    baggingAlgo()
