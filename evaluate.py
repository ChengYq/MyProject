import sys

# filePath = sys.argv[1]

# filePathPredit = '/home/chyq/Document/MyProject/DataSet/MDP/my/my_kc1_origin.predict'
# filePathPredit = '/home/chyq/Document/MyProject/DataSet/MDP/my/my_kc1_corr.predict'
# filePathPredit = '/home/chyq/Document/MyProject/DataSet/MDP/my/my_kc1_info.predict'


# filePathPredit = '/home/chyq/Document/MyProject/DataSet/MDP/my/my_kc1_origin.predict_w'
# filePathPredit = '/home/chyq/Document/MyProject/DataSet/MDP/my/my_kc1_corr.predict_w'
# filePathPredit = '/home/chyq/Document/MyProject/DataSet/MDP/my/my_kc1_info.predict_w'


# filePathPredit = '/home/chyq/Document/MyProject/DataSet/MDP/my/my_kc1_origin.predict_noise_w'
# filePathPredit = '/home/chyq/Document/MyProject/DataSet/MDP/my/my_kc1_corr.predict_noise_w'
# filePathPredit = '/home/chyq/Document/MyProject/DataSet/MDP/my/my_kc1_info.predict_noise_w'

filePathTest = '/home/chyq/Document/MyProject/DataSet/MDP/my/my_kc1_test'

try:
    f = open(filePathPredit, 'r')
    lines = f.readlines()
finally:
    f.close()

y_pred = []
for i in lines:
    y_pred.append(i.strip('\n'))

y_pred = map(float, y_pred)

try:
    ff = open(filePathTest, 'r')
    lines = ff.readlines()
finally:
    ff.close()

y_test = []
for i in lines:
    dd = i.split(' ')
    y_test.append(dd[0].strip('\n'))

y_test = map(float, y_test)

from sklearn.metrics import classification_report

r = classification_report(y_test, y_pred)
print r
