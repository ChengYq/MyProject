# coding=utf-8
def bagIt(dataSet):
    # dataSet为List类型的
    noDefectCount = 0
    defectCount = 0
    for i in dataSet:

        if i[-1].lower() in ['true', 'y', 'yes']:
            defectCount += 1
        elif i[-1].lower() in ['false', 'n', 'no']:
            noDefectCount += 1

    print len(dataSet)
    print noDefectCount, defectCount


if __name__ == '__main__':
    from PreProcess import createDataset
    from os import path

    filePath = path.abspath(path.join(path.dirname(__file__), path.pardir, r'DataSet', r'MDP', r'PROMISE', r'cm1.arff'))
    train, test = createDataset.createDataSet(filePath, 5)
    bagIt(train)
