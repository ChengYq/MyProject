import sys

# filePath = sys.argv[1]
filePathPredit = '/home/chyq/Download/libsvm-3.22/tools/my_kc1.test.predict'
filePathTest = '/home/chyq/Download/libsvm-3.22/tools/my_kc1.test.scale'

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
