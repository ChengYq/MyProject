import sys
import random

if (len(sys.argv) < 3):
    print("usage: ./python convertToSVM.py inputfilename train test")
# print sys.argv[1]

inputfilename = sys.argv[1]
fin = open(inputfilename, 'r')
lines = fin.readlines()
fin.close()
outputfilename1 = sys.argv[2]
train1 = open(outputfilename1, 'w')
outputfilename2 = sys.argv[3]
test1 = open(outputfilename2, 'w')
seq = range(100)

beginToRead = False
for line in lines:
    if beginToRead == True:
        if len(line) > 5:  # not an empty line
            # read this line
            dataList = line.split(',')
            resultLine = ''
            label = dataList[-1].strip()
            if label.lower() in ['yes', 'y', 'true']:
                toAppend = '1'
            else:
                toAppend = '-1'
            resultLine += toAppend
            resultLine += ' '
            for i in range(1, len(dataList) - 1):
                resultLine += str(i)
                resultLine += (":" + dataList[i] + " ")
            # print(resultLine)
            if random.choice(seq) < 20:
                test1.write(resultLine + "\n")
            else:
                train1.write(resultLine + "\n")

    if line[0:5] == '@data':
        beginToRead = True

train1.close()
test1.close()
