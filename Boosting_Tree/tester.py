"""
Programmer  :   EOF
Date        :   2015.11.22
File        :   tester.py

File Description:
    This file is used to test the adaboost which is a classical
automatic classifier.

"""

import numpy
import matplotlib.pyplot as pyplot
from boostingTree import BoostingTree

"""
Demo One
"""
Original_Data = numpy.array([
[2],
[3],
[4],
[5],
[6],
[7],
[8],
[9],
[10],
[1]
]).transpose()

ExpVal = numpy.array([
[5.70],
[5.91],
[6.40],
[6.80],
[7.05],
[8.90],
[8.70],
[9.00],
[9.05],
[5.56]
]).transpose()

ExpVal = ExpVal.flatten()

a = BoostingTree(Original_Data, ExpVal)

a.train(100)

Unknow = numpy.array([
[12],
[11],
[1]
]).transpose()

print "Test case input:", Unknow
print "Predict the output :", a.prediction(Unknow)

