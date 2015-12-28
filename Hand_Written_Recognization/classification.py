"""
Programmer  :   EOF
File        :   .py
Date        :   2015.12.28
E-mail      :   jasonleaster@163.com
"""
import numpy

from sklearn.ensemble       import AdaBoostClassifier
from sklearn.tree           import DecisionTreeClassifier
from initia_training_set    import ImageSet

trainingset = ImageSet("./train-images.idx3-ubyte",
                       "./train-labels.idx1-ubyte")

testset     = ImageSet("./t10k-images.idx3-ubyte",
                       "./t10k-labels.idx1-ubyte")

bdt = AdaBoostClassifier(DecisionTreeClassifier(max_depth = 6),
                        n_estimators = 500,
                        learning_rate = 1)

trainingImgMat = [None for i in range(trainingset.SampleNum)]
trainingLabel  = [trainingset.labels[i] for i in range(trainingset.SampleNum)]

testImgMat = [None for i in range(testset.SampleNum)]
testLabel  = [testset.labels[i] for i in range(testset.SampleNum)]


for i in range(trainingset.SampleNum):
    trainingImgMat[i] = trainingset.flattenImg(i)

for i in range(testset.SampleNum):
    testImgMat[i]     = testset.flattenImg(i)

bdt.fit(trainingImgMat, trainingLabel)

print bdt.estimator_errors_

output =  bdt.predict(testImgMat)

print "Correct: ", 100 * numpy.count_nonzero(output == testLabel)/1000. , "%"
