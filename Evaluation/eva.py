# coding=utf-8
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report


def myConvert(x):
    if x is 'true':
        return 1
    elif x is 'false':
        return -1


def evaluationRes(y_pred, y_actual):
    # 本函数通吃所有的情况
    # 全部转换为true___1     false____-1
    y_pred_label = []
    y_actual_label = []
    if (not isinstance(y_pred[0], list)) and (isinstance(y_pred[0], str)):
        # 如果输入的只是label，而不带有前边的feature部分,而且已经确定label是true false这种样子的
        y_pred_label = map(myConvert, y_pred)

    elif isinstance(y_pred[0], list):
        # 如果输入的是带有feature的数据集，而不是仅仅只有Label
        for i in y_pred:
            if i[-1] == 'true':
                y_pred_label.append(1.0)
            elif i[-1] == 'false':
                y_pred_label.append(-1.0)
            else:
                raise Exception('not true nor false')
    else:
        # 这种情况就是输入的是：不带有feature，而且label是+1 -1的情况
        y_pred_label = y_pred

    if (not isinstance(y_actual[0], list)) and (isinstance(y_actual[0], str)):
        # 如果输入的只是label，而不带有前边的feature部分,而且已经确定label是true false这种样子的
        y_actual_label = map(myConvert, y_actual)

    elif isinstance(y_actual[0], list):
        # 如果输入的是带有feature的数据集，而不是仅仅只有Label
        for i in y_actual:
            # print i

            if i[-1] == 'true':
                y_actual_label.append(1.0)
            elif i[-1] == 'false':
                y_actual_label.append(-1.0)
            else:
                raise Exception('not true nor false')

    else:
        # 这种情况就是输入的是：不带有feature，而且label是+1 -1的情况
        y_actual_label = y_actual

    confusionMatrix = confusion_matrix(y_actual_label, y_pred_label)
    report = classification_report(y_actual_label, y_pred_label)

    return confusionMatrix, report
