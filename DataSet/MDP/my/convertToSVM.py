import sys

inputfilename = sys.argv[1]
fin = open(inputfilename, 'r')
lines = fin.readlines()
fin.close()
outputfilename1 = sys.argv[2]
res = open(outputfilename1, 'w')


beginToRead = False
for line in lines:
    if beginToRead == True:
        if len(line) > 5:  # not an empty line
            # read this line
            dataList = line.split(',')
            resultLine = ''
            label = dataList[-1].strip()
            if label.lower() in ['yes', 'y', 'true', '1.0']:
                toAppend = '1'
            else:
                toAppend = '-1'
            resultLine += toAppend
            resultLine += ' '
            for i in range(1, len(dataList) - 1):
                resultLine += str(i)
                resultLine += (":" + dataList[i] + " ")
            # print(resultLine)
            res.write(resultLine + "\n")

    if line[0:5].lower() == '@data':
        beginToRead = True

res.close()
