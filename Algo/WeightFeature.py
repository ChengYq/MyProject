# coding=utf-8
# 实现的是feature space的模糊隶属计算

# coding=utf-8


# f(x), x∈[lower_bound, upper_bound]
# x = lower_bound + decimal(chromosome)×(upper_bound-lower_bound)/(2^chromosome_size-1)

import random
from Algo.libsvmWeight.python.svmutil import *
from sklearn.metrics import f1_score
import math
import numpy as np


def rbf(x, y, g):
    a = np.linalg.norm(x - y)
    b = 2 * g * g
    return np.exp(-a * a / b)


def distance(trainset, g):
    # 输入的是训练集
    # 返回的是一个数组
    second = 0
    count = len(trainset)
    result = []
    for k in range(count):
        for j in range(count):
            # print trainset[k], trainset[j]
            second += rbf(trainset[k], trainset[j], g)

    for i in range(count):
        first = 0
        for j in range(count):
            first += rbf(trainset[i], trainset[j], g)

        a = rbf(trainset[i], trainset[i], g)
        b = (2.0 / count) * first
        c = (1.0 / (count ** 2)) * second
        temp = a - b + c
        result.append(temp)

    return result


def create_weight(train_feature, train_label, g):
    train_feature_pos = []
    train_feature_neg = []
    noise_weight = []
    for i in range(len(train_label)):
        if train_label[i] == 1.0:
            train_feature_pos.append(train_feature[i, :])
        else:
            train_feature_neg.append(train_feature[i, :])

    pos_count = len(train_feature_pos)
    neg_count = len(train_feature_neg)

    imbalance_ratio = float(pos_count) / float(neg_count)

    pos_dis = distance(train_feature_pos, g)
    neg_dis = distance(train_feature_neg, g)

    radius_pos = max(pos_dis)
    radius_neg = max(neg_dis)

    ii = 0
    iii = 0
    for i in range(len(train_label)):
        if train_label[i] == 1.0:
            f = 1 - np.sqrt(float(pos_dis[ii]) / float(radius_pos))
            ii += 1
        else:
            f = 1 - np.sqrt(float(neg_dis[iii]) / float(radius_neg))
            f *= imbalance_ratio
            iii += 1

        # print f
        noise_weight.append(f)
    return noise_weight


def b2d(b):  # 将二进制转化为十进制 x∈[0,10]
    tempC = int(''.join(str(e) for e in b[:13]), 2)
    tempg = int(''.join(str(e) for e in b[13:]), 2)

    c = 1 + tempC * (8000 - 1) / (2 ** 13 - 1)
    g = 0.01 + tempg * (1 - 0.01) / (2 ** 10 - 1)

    return c, g


def best(pop, fitvalue):  # 找出适应函数值中最大值，和对应的个体
    px = len(pop)
    bestindividual = pop[0]
    bestfit = fitvalue[0]
    for i in range(1, px):
        if (fitvalue[i] > bestfit):
            bestfit = fitvalue[i]
            bestindividual = pop[i]
    return [bestindividual, bestfit]


def decodechrom(pop):  # 将种群的二进制基因转化为十进制（0,1023）
    temp = []
    for i in range(len(pop)):
        t = 0
        for j in range(35):
            t += pop[i][j] * (math.pow(2, j))
        temp.append(t)
    return temp


def calobjvalue(pop, y, x, yt, xt, featured_trainset, train_label):  # 计算目标函数值
    tempC = []
    tempg = []
    objvalue = []
    for i in range(len(pop)):
        tempC.append(int(''.join(str(e) for e in pop[i][:13]), 2))
        tempg.append(int(''.join(str(e) for e in pop[i][13:]), 2))
    for i in range(len(tempC)):
        c = 1 + tempC[i] * (8000 - 1) / (2 ** 13 - 1)
        g = 0.01 + tempg[i] * (1 - 0.01) / (2 ** 10 - 1)
        print '-------------'
        print c, g

        W = create_weight(featured_trainset, train_label, g)
        prob = svm_problem(W, y, x)
        p = '-c {0} -g {1}'.format(c, g)
        para = svm_parameter(p)
        model = svm_train(prob, para)
        p_label, p_acc, p_val = svm_predict(yt, xt, model)
        p_label = np.array(p_label)
        nF = f1_score(yt, p_label)
        if np.max(p_label) == np.min(p_label):
            print 'yes,it is all equal!'
            nF = nF / 5.0
            # nF=0.0
        print nF
        objvalue.append(nF)
    return objvalue  # 目标函数值objvalue[m] 与个体基因 pop[m] 对应


def crossover(pop, pc):  # 个体间交叉，实现基因交换
    poplen = len(pop)
    i = 0
    while i < poplen:
        # for i in range(poplen - 1):
        if (random.random() < pc):
            cpoint = random.randint(0, len(pop[0]))
            temp1 = []
            temp2 = []
            temp1.extend(pop[i][0: cpoint])
            temp1.extend(pop[i + 1][cpoint: len(pop[i])])
            temp2.extend(pop[i + 1][0: cpoint])
            temp2.extend(pop[i][cpoint: len(pop[i])])
            pop[i] = temp1
            pop[i + 1] = temp2
        i += 2
    return pop


def mutation(pop, pm):  # 基因突变
    px = len(pop)
    py = len(pop[0])

    for i in range(px):
        if (random.random() < pm):
            mpoint = random.randint(0, py - 1)
            if (pop[i][mpoint] == 1):
                pop[i][mpoint] = 0
            else:
                pop[i][mpoint] = 1
    return pop


def sum(fitvalue):
    total = 0
    for i in range(len(fitvalue)):
        total += fitvalue[i]
    return total


def cumsum(fitvalue):
    # 计算累计概率
    newfitvalue = [0 for i in range(len(fitvalue))]
    for i in range(len(fitvalue)):
        t = 0
        j = 0
        while (j <= i):
            t += fitvalue[j]
            j = j + 1
        newfitvalue[i] = t
    return newfitvalue


def selection(pop, fitvalue):
    # 自然选择（轮盘赌算法）
    newfitvalue = []
    totalfit = sum(fitvalue)
    for i in range(len(fitvalue)):
        newfitvalue.append(fitvalue[i] / totalfit)
    newfitvalue = cumsum(newfitvalue)
    ms = []
    poplen = len(pop)
    for i in range(poplen):
        ms.append(random.random())  # random float list ms
    ms.sort()
    fitin = 0
    newin = 0
    newpop = pop
    while newin < poplen:
        if (ms[newin] < newfitvalue[fitin]):
            newpop[newin] = pop[fitin]
            newin = newin + 1
        else:
            fitin = fitin + 1
    select_pop = newpop
    return select_pop


def geneticFeature(x, y, xt, yt, featured_trainset, train_label):
    popsize = 2  # 种群的大小
    # 用遗传算法求函数最大值：
    # f(x)=10*sin(5x)+7*cos(4x) x∈[0,10]

    chromlength = 23  # 基因片段的长度 13+10=23
    pc = 0.6  # 两个个体交叉的概率
    pm = 0.01  # 基因突变的概率
    results = []
    temp = []
    pop = []

    for i in range(popsize):
        for j in range(chromlength):
            temp.append(random.randint(0, 1))
        pop.append(temp)
        temp = []
    for i in range(2):  # 繁殖100代
        print '$$$$$$$$$$$$$$$'
        print i
        print '$$$$$$$$$$$$$$$'

        objvalue = calobjvalue(pop, y, x, yt, xt, featured_trainset, train_label)  # 计算目标函数值
        [bestindividual, bestfit] = best(pop, objvalue)  # 选出最好的个体和最好的函数值
        results.append([bestfit, b2d(bestindividual)])  # 每次繁殖，将最好的结果记录下来
        select_pop = selection(pop, objvalue)  # 自然选择，淘汰掉一部分适应性低的个体
        cross_pop = crossover(select_pop, pc)  # 交叉繁殖
        pop = mutation(cross_pop, pm)  # 基因突变

    results.sort()
    print results
    print(results[-1])  # 打印函数最大值和对应的
    return results[-1]


def test():
    from PreProcess.createDataset import createDataSet
    from os import path
    from FeatureSelection import FeatureSelectionProcess
    from PreProcess.minmax2 import minmaxscaler
    from PreProcess.createDataset import featureAndLabel
    from Fsvmcil import create_weight
    import arff

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

    # W=create_weight(featured_trainset, train_label)
    best_para = geneticFeature(x, y, xtest, ytest, featured_trainset, train_label)

    print best_para


if __name__ == '__main__':
    test()
