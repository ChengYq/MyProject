# coding= utf-8
# 参考： http://www.techv5.com/topic/289/
# 参考： http://www.csie.ntu.edu.tw/~cjlin/libsvm/faq.html#/Q04:_Training_and_prediction

# 2 这个版本是为了生成arff文件的！
# 我的目的是： 自己bagging+ReliefF后的数据集生成arff, 然后把它放进easy.py进行训练。先看看我的方法有没有进步。
# 我自己的方法：参看自己的说明文档：    /home/chyq/Document/MyProject/DataSet/MDP
# 验证目的：我的Bagging+ReliefF方法到底有没有用。因此输出arff，然后用easy.py计算


# 生成以下文件
# 1. 原始的训练集、测试集
# 2. bagging+ReliefF的训练集、测试集
# 3. 除去Correlation的训练集、测试集
# 4. Bagging+Correlation+Information(ratio)的训练集、测试集
# 5. 生成noiseWeight,用来消除outliner的影响

# 然后放进 easy.py 训练
import numpy as np

def test():
    from PreProcess.createDataset import createDataSet
    from os import path
    from FeatureSelection import FeatureSelectionProcess
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

    # from sklearn import preprocessing
    # trainset = preprocessing.scale(train_feature)
    # testset = preprocessing.scale(test_feature)


    ############################
    # 1.
    # 生成原始的arff,包括训练集、测试集

    origin_train = np.c_[trainset, train_label]
    testData = np.c_[testset, test_label]

    arff_obj = {'relation': relation, 'attributes': attribute, 'data': origin_train}
    import arff
    tr0 = arff.dumps(arff_obj)
    try:
        f1 = open('/home/chyq/Document/MyProject/DataSet/MDP/my/my_kc1_origin.arff', 'w')
        f1.write(tr0)
    finally:
        f1.close()

    arff_obj = {'relation': relation, 'attributes': attribute, 'data': testData}
    te = arff.dumps(arff_obj)
    try:
        f2 = open('/home/chyq/Document/MyProject/DataSet/MDP/my/my_kc1_test.arff', 'w')
        f2.write(te)
    finally:
        f2.close()


    #############################

    2.
    # 生成Bagging+Correlation 的特征子集，需要注意的是，attribute已经经过了处理。数量和featureed_trainset是一样的
    featured_trainset, featured_attribute = FeatureSelectionProcess.selectedSet(trainset, train_label, attribute,
                                                                                trainset)
    print "bagging+corr", len(featured_attribute)

    arff_obj = {'relation': relation, 'attributes': featured_attribute, 'data': featured_trainset}

    # 写入to1
    to1 = arff.dumps(arff_obj)
    try:
        f = open('/home/chyq/Document/MyProject/DataSet/MDP/my/my_kc1_featured.arff', 'w')
        f.write(to1)
    finally:
        f.close()


    #############################
    # 3.
    # 下边这里的代码：是用于检索出那些重复的下标，放进toDelete内
    # 事实证明，是否经过了featureSelection，对计算Corr几乎没有影响

    from FeatureSelection import correlation
    toDelete = correlation.corr(trainset)

    Corr_attribute = []
    for i in range(len(attribute)):
        if i not in toDelete:
            Corr_attribute.append(attribute[i])

    print "corr len", len(Corr_attribute)
    t = np.c_[trainset, train_label]
    noCorr_data = np.delete(t, toDelete, axis=1)

    arff_obj = {'relation': relation, 'attributes': Corr_attribute, 'data': noCorr_data}

    # arff_obj = {'relation': relation, 'attributes': attribute, 'data': featured_trainset}

    to2 = arff.dumps(arff_obj)
    try:
        f = open('/home/chyq/Document/MyProject/DataSet/MDP/my/my_kc1_corr.arff', 'w')
        f.write(to2)
    finally:
        f.close()

    ####################
    # 4.
    from FeatureSelection import afterFeatureSelection3
    featured_trainset, featured_attribute = afterFeatureSelection3.selectedSet(trainset, train_label, attribute,
                                                                               train_feature)

    arff_obj = {'relation': relation, 'attributes': featured_attribute, 'data': featured_trainset}

    print "info_gain", len(featured_attribute)
    # 写入to3
    to3 = arff.dumps(arff_obj)
    try:
        f = open('/home/chyq/Document/MyProject/DataSet/MDP/my/my_kc1_info.arff', 'w')
        f.write(to3)
    finally:
        f.close()


        ######
        # 5
        # 写入一个文件，方便读取
    from Algo import Fsvmcil
    Fsvmcil.create_weight(trainset, train_label)


if __name__ == '__main__':
    test()