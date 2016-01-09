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
from adaboost import AdaBoost

"""
Samples for AdaBoost
"""
Original_Data = numpy.array([
[1.85,  1.95],
[3.15,  1.7],
[4,     2.7],
[3.75,  3.95],
[2.8,   4.4],
[2.35,  3.2],
[3.05,  2.25],
[3.55,  2.6],
[3.1,   3],
[3,     3.4],
[1,     7.3],
[1.4,   6.7],
[3.05,  6.9],
[4.3,   7.15],
[4.75,  7],
[5.5,   5.85],
[5.95,  4.75],
[6.45,  3.15],
[6.5,   1.35],
[6.3,   0.95],
[5.95,  0.85],
[5.95,  1.6],
[5.85,  2.75],
[5.65,  4],
[5.35,  5.25],
[5,     6.15],
[4.7,   6.3],
[3.85,  6.5],
[2.55,  6.55],
[1.4,   6.65],
[0.6,   6.75],
[0.6,   6.85],
[0.8,   1.00]]).transpose()

Tag = numpy.array([
[-1],
[-1],
[-1],
[-1],
[-1],
[-1],
[-1],
[-1],
[-1],
[-1],
[+1],
[+1],
[+1],
[+1],
[+1],
[+1],
[+1],
[+1],
[+1],
[+1],
[+1],
[+1],
[+1],
[+1],
[+1],
[+1],
[+1],
[+1],
[+1],
[+1],
[+1],
[+1],
[+1]]).transpose()

Tag = Tag.flatten()

for i in range(len(Tag)):
    if Tag[i] == 1:
        pyplot.plot(Original_Data[0][i], Original_Data[1][i], \
                    '+r', markersize = 10)
    else:
        pyplot.plot(Original_Data[0][i], Original_Data[1][i], \
                    '+b', markersize = 10)




a = AdaBoost(Original_Data, Tag)

a.train(100)

TestCase = [[0.55, 1.1, 5.35], 
            [4.4,  2.8, 0.9]]

output = a.prediction(TestCase)

for i in range(len(output)):
    if output[i] == 1:
        pyplot.plot(TestCase[0][i], TestCase[1][i], \
                    'or', markersize = 20)
    else:
        pyplot.plot(TestCase[0][i], TestCase[1][i], \
                    'ob', markersize = 20)

pyplot.show()
