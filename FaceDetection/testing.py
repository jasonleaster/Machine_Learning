"""
Programmer  :   EOF
File        :   testing.py
Date        :   2015.12.29
E-mail      :   jasonleaster@163.com

Description :

"""
from image import ImageSet
from matplotlib import pyplot
import numpy
import os
from config import *
from adaboost import AdaBoost

fileObj = open(FEATURE_FILE_TESTING, "a+")

# if that is a empty file
if os.stat(FEATURE_FILE_TESTING).st_size == 0:

    print "First time to load the testing set ..."
    TestSetFace          = ImageSet(TEST_FACE, sampleNum = TEST_SAMPLE)
    TestSetNonFace       = ImageSet(TEST_NONFACE, sampleNum = TEST_SAMPLE)

    Original_Data_Face = [
        [sum(TestSetFace.images[i].haarA),
         sum(TestSetFace.images[i].haarB),
         sum(TestSetFace.images[i].haarC),
         sum(TestSetFace.images[i].haarD)]
        for i in range(TestSetFace.sampleNum)
        ]

    Original_Data_NonFace =[ 
        [sum(TestSetNonFace.images[i].haarA),
         sum(TestSetNonFace.images[i].haarB),
         sum(TestSetNonFace.images[i].haarC),
         sum(TestSetNonFace.images[i].haarD)]
        for i in range(TestSetNonFace.sampleNum)
        ]

    Original_Data = numpy.array(Original_Data_Face + \
                        Original_Data_NonFace).transpose()

    for i in range(Original_Data.shape[0]):
        for j in range(Original_Data.shape[1]):
            fileObj.write(str(Original_Data[i][j]) + "\n")

    fileObj.flush()
else:
    print "Haar features have been calculated."
    print "Loading features ..."

    tmp = fileObj.readlines()

    Original_Data = []
    for i in range(0, len(tmp), FEATURE_TYPE_NUM):
        haarGroup = []
        for j in range(i, i + FEATURE_TYPE_NUM):
            haarGroup.append(float(tmp[j]))

        Original_Data.append(haarGroup)

    Original_Data = numpy.array(Original_Data).transpose()

fileObj.close()

fileObj = open(ADABOOST_FILE, "a+")
print "Constructing AdaBoost from existed model data"

a = AdaBoost(train = False)

tmp = fileObj.readlines()

for i in range(0, len(tmp), 4):

    alpha, demention, label, threshold = None, None, None, None

    for j in range(i, i + 4):
        if (j % 4) == 0:
            alpha = float(tmp[j])
        elif (j % 4) == 1:
            demention = int(tmp[j])
        elif (j % 4) == 2:
            label = float(tmp[j])
        elif (j % 4) == 3:
            threshold = float(tmp[j])

    classifier = a.Weaker(train = False)
    classifier.constructor(demention, label, threshold)
    a.G[i/4] = classifier
    a.alpha[i/4] = alpha
    a.N += 1

print "Construction finished"
fileObj.close()

output = a.prediction(Original_Data)

print numpy.count_nonzero(output[0:TEST_SAMPLE] > 0) * 1./ TEST_SAMPLE

print numpy.count_nonzero(output[TEST_SAMPLE:TEST_SAMPLE*2] < 0) * 1./ TEST_SAMPLE
