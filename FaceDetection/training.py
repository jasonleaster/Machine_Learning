"""
Programmer  :   EOF
File        :   tester.py
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

fileObj = open(FEATURE_FILE_TRAINING, "a+")

# if that is a empty file
if os.stat(FEATURE_FILE_TRAINING).st_size == 0:

    print "First time to load the training set ..."
    TrainingSetFace      = ImageSet(TRAINING_FACE, sampleNum = POSITIVE_SAMPLE)
    TrainingSetNonFace   = ImageSet(TRAINING_NONFACE, sampleNum = NEGATIVE_SAMPLE)

    Original_Data_Face = [
         sum(TrainingSetFace.images[i].haarA + 
         TrainingSetFace.images[i].haarB + 
         TrainingSetFace.images[i].haarC +
         TrainingSetFace.images[i].haarD
        for i in range(TrainingSetFace.sampleNum)
        ]

    Original_Data_NonFace =[ 
        [sum(TrainingSetNonFace.images[i].haarA),
         sum(TrainingSetNonFace.images[i].haarB),
         sum(TrainingSetNonFace.images[i].haarC),
         sum(TrainingSetNonFace.images[i].haarD)]
        for i in range(TrainingSetNonFace.sampleNum)
        ]

    Original_Data = numpy.array(Original_Data_Face + \
                        Original_Data_NonFace).transpose()

    assert Original_Data.shape[0] == FEATURE_TYPE_NUM

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

SampleDem = Original_Data.shape[0]
SampleNum = Original_Data.shape[1]

assert SampleNum == (POSITIVE_SAMPLE + NEGATIVE_SAMPLE)

Label_Face    = [+1 for i in range(POSITIVE_SAMPLE)]
Label_NonFace = [-1 for i in range(NEGATIVE_SAMPLE)]

Label = numpy.array(Label_Face + Label_NonFace)

a = AdaBoost(Original_Data, Label)

a.train(5000)

fileObj = open(ADABOOST_FILE, "a+")

for m in range(a.N):
    fileObj.write(str(a.alpha[m]) + "\n")
    fileObj.write(str(a.G[m].opt_demention) + "\n")
    fileObj.write(str(a.G[m].opt_label) + "\n")
    fileObj.write(str(a.G[m].opt_threshold) + "\n")

fileObj.flush()
fileObj.close()
