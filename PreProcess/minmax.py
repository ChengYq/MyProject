# coding=utf-8
def minmaxscaler(feature, lower=0, upper=1, record='n'):
    import numpy as np
    x_feature = np.array(feature)

    if record is 'n':

        x_feature_min = x_feature.min(axis=0)
        x_feature_max = x_feature.max(axis=0)

        try:
            f = open('/home/chyq/Document/MyProject/PreProcess/max_min_config.txt', 'w')
            f.write(str(x_feature_min) + '\r\n' + str(x_feature_max))

            for i in range(x_feature.shape[1]):
                if i == 0:
                    x_scaled = (x_feature[:, i] - x_feature_min[i]) / float(x_feature_max[i] - x_feature_min[i]) * (
                        upper - lower) + lower
                else:
                    to_insert = (x_feature[:, i] - x_feature_min[i]) / float(x_feature_max[i] - x_feature_min[i]) * (
                        upper - lower) + lower
                    x_scaled = np.c_[x_scaled, to_insert]
        finally:
            if f:
                f.close()

    if record is 'y':
        try:
            f = open('/home/chyq/Document/MyProject/PreProcess/max_min_config.txt', 'r')
            lineNum = 0
            for i in f.readlines():
                if lineNum == 0:
                    x_feature_min = list(i)
                    lineNum += 1
                elif lineNum == 1:
                    x_feature_max = list(i)

            for i in range(x_feature.shape[1]):
                if i == 0:
                    x_scaled = (x_feature[:, i] - x_feature_min[i]) / float(x_feature_max[i] - x_feature_min[i]) * (
                        upper - lower) + lower
                else:
                    to_insert = (x_feature[:, i] - x_feature_min[i]) / float(x_feature_max[i] - x_feature_min[i]) * (
                        upper - lower) + lower
                    x_scaled = np.c_[x_scaled, to_insert]

        finally:
            if f:
                f.close()

    print x_scaled
    return x_scaled
