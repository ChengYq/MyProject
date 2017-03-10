# coding = utf-8

from sklearn.feature_selection import mutual_info_classif


def mutual(features, labels):
    a = mutual_info_classif(features, labels, discrete_features=True)
    return a
