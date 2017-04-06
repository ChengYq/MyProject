# coding=utf-8
import numpy as np


def create_weight(train_feature, train_label):
    # 输入 train_feature, train_label 都是numpy类型的
    # 输出：noise_weight 一个list，反映了权值
    # 参考： FSVM-CIL，exp的方法不好用，因为我的值都太大
    train_feature_pos = []
    train_feature_neg = []
    norm_pos = []
    norm_neg = []
    norm_total = []
    noise_weight = []
    for i in range(len(train_label)):
        if train_label[i] == 1.0:
            train_feature_pos.append(train_feature[i, :])
        else:
            train_feature_neg.append(train_feature[i, :])

    train_feature_pos_mean = np.mean(train_feature_pos, axis=0)
    train_feature_neg_mean = np.mean(train_feature_neg, axis=0)

    for i in range(len(train_label)):
        if train_label[i] == 1.0:
            temp = np.linalg.norm(train_feature[i, :] - train_feature_pos_mean)
            norm_pos.append(temp)
        else:
            temp = np.linalg.norm(train_feature[i, :] - train_feature_neg_mean)
            norm_neg.append(temp)
        norm_total.append(temp)

    print norm_total

    max_pos = max(norm_pos)
    max_neg = max(norm_neg)
    print max_neg
    for i in range(len(train_label)):
        if train_label[i] == 1.0:
            f = 1 - (float(norm_total[i]) / float(max_pos))
        else:
            f = 1 - (float(norm_total[i]) / float(max_neg))

        print f
        noise_weight.append(f)

    try:
        f = open(r'/home/chyq/Document/MyProject/DataSet/MDP/my/weight.txt', 'w')
        weight = [x + '\n' for x in map(str, noise_weight)]
        f.writelines(weight)
    finally:
        f.close()

    return f


def test():
    from PreProcess.createDataset import createDataSet
    from os import path
    from FeatureSelection import afterFeatureSelection2
    from PreProcess.minmax2 import minmaxscaler
    from PreProcess.createDataset import featureAndLabel

    filePath = path.abspath(path.join(path.dirname(__file__), path.pardir, r'DataSet', r'MDP', r'D1', r'KC1.arff'))

    data, trainsetWithLabel, testsetWithLabel, relation, attribute = createDataSet(filePath, 5)

    # featureSet = bagging.bagIt(trainset)

    # 分离出训练集、测试集的feature, label
    train_feature, train_label = featureAndLabel(trainsetWithLabel)
    test_feature, test_label = featureAndLabel(testsetWithLabel)

    # 做normalization 得出的结果在trainset, test。
    trainset, x_min, x_max = minmaxscaler(train_feature)
    testset = minmaxscaler(test_feature, x_feature_min=x_min, x_feature_max=x_max)

    file = create_weight(train_feature, train_label)


if __name__ == '__main__':
    test()
