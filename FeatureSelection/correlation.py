# coding=utf-8
import numpy as np


def corr(feature):
    toDelete = []
    corr_matrix = np.corrcoef(feature, rowvar=0)
    row, col = corr_matrix.shape
    for i in range(row):
        for j in range(i + 1, col):
            if (i not in toDelete) and (j not in toDelete):
                if corr_matrix[i, j] > 0.95:
                    toDelete.append(j)
    return toDelete
