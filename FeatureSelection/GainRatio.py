# coding=utf-8
import numpy as np


def calc_ent(x):
    # x numpy 类型
    # 计算熵
    x_value_list = set([x[i] for i in range(x.shape[0])])
    ent = 0.0
    for x_value in x_value_list:
        p = float(x[x == x_value].shape[0]) / x.shape[0]
        logp = np.log2(p)
        ent -= p * logp

    return ent


def calc_condition_ent(x, y):
    # 计算条件熵
    # calc ent(y|x)
    x_value_list = set([x[i] for i in range(x.shape[0])])
    ent = 0.0
    for x_value in x_value_list:
        sub_y = y[x == x_value]
        temp_ent = calc_ent(sub_y)
        ent += (float(sub_y.shape[0]) / y.shape[0]) * temp_ent

    return ent


def calc_inf_gain(x, y):
    # 计算信息增益比
    # 若取消掉除以base_ent， 则计算结果是信息增益

    base_ent = calc_ent(y)
    condition_ent = calc_condition_ent(x, y)
    ent_grap = base_ent - condition_ent

    return ent_grap


def calc_condition_ent_ratio(x, y):
    # 计算条件熵
    # calc ent(y|x)
    x_value_list = set([x[i] for i in range(x.shape[0])])
    ent = 0.0
    split_info = 0.0
    for x_value in x_value_list:
        sub_y = y[x == x_value]
        temp_ent = calc_ent(sub_y)
        ent += (float(sub_y.shape[0]) / y.shape[0]) * temp_ent
        split_info -= (float(sub_y.shape[0]) / y.shape[0]) * np.log2(float(sub_y.shape[0]) / y.shape[0])

    return split_info


def calc_gain_ratio(x, y):
    # 计算信息增益比
    # 若取消掉除以base_ent， 则计算结果是信息增益

    base_ent = calc_ent(y)
    condition_ent = calc_condition_ent(x, y)
    ratio = (base_ent - condition_ent) / calc_condition_ent_ratio(x, y)

    return ratio


def information_gain(trainset, train_label):
    """

    :rtype: info_gain,gain_ratio
    """
    info_gain = []
    gain_ratio = []

    for i in range(trainset.shape[1]):
        x = trainset[:, i]
        info_gain.append(calc_inf_gain(x, train_label))
        gain_ratio.append(calc_gain_ratio(x, train_label))

    info_gain = sorted(zip(info_gain, range(len(info_gain))), key=lambda x: x[0])
    gain_ratio = sorted(zip(gain_ratio, range(len(gain_ratio))), key=lambda x: x[0])

    return info_gain, gain_ratio


def test():
    from PreProcess.createDataset import createDataSet
    from os import path
    from PreProcess.minmax2 import minmaxscaler
    from PreProcess.createDataset import featureAndLabel

    filePath = path.abspath(path.join(path.dirname(__file__), path.pardir, r'DataSet', r'MDP', r'D1', r'KC1.arff'))

    data, trainsetWithLabel, testsetWithLabel, relation, attribute = createDataSet(filePath, 10)

    # featureSet = bagging.bagIt(trainset)

    # 分离出训练集、测试集的feature, label
    train_feature, train_label = featureAndLabel(trainsetWithLabel)
    test_feature, test_label = featureAndLabel(testsetWithLabel)

    # 做normalization 得出的结果在trainset, test。
    trainset, x_min, x_max = minmaxscaler(train_feature)
    testset = minmaxscaler(test_feature, x_feature_min=x_min, x_feature_max=x_max)


if __name__ == '__main__':
    test()
