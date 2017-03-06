# coding=utf-8
def minmaxscaler(feature, lower=0, upper=1, x_feature_min=None, x_feature_max=None):
    import numpy as np
    x_feature = np.array(feature)

    if (x_feature_min is None) and (x_feature_max is None):

        x_feature_min = x_feature.min(axis=0)
        x_feature_max = x_feature.max(axis=0)

        for i in range(x_feature.shape[1]):

            if i == 0:
                x_scaled = (x_feature[:, i] - x_feature_min[i]) / float(x_feature_max[i] - x_feature_min[i]) * (
                    upper - lower) + lower
            else:
                to_insert = (x_feature[:, i] - x_feature_min[i]) / float(x_feature_max[i] - x_feature_min[i]) * (
                    upper - lower) + lower
                x_scaled = np.c_[x_scaled, to_insert]

        return x_scaled, x_feature_min, x_feature_max

    else:

        for i in range(x_feature.shape[1]):
            if i == 0:
                x_scaled = (x_feature[:, i] - x_feature_min[i]) / float(x_feature_max[i] - x_feature_min[i]) * (
                    upper - lower) + lower
            else:
                to_insert = (x_feature[:, i] - x_feature_min[i]) / float(x_feature_max[i] - x_feature_min[i]) * (
                    upper - lower) + lower
                x_scaled = np.c_[x_scaled, to_insert]

        # print x_scaled
        return x_scaled
