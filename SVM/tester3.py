"""
Programmer  : EOF
Date        : 2015.11.23
File        : test.py
E-mail      : jasonleaster@163.com

"""
import numpy
import matplotlib.pyplot as pyplot
from svm import SVM

Original_Data = numpy.array([
    [3, 3],
    [4, 3],
    [1, 1],
    [0, 0],
    [6, 6],
    [1, 2.2]]).transpose()

Tag = numpy.array([
    [+1],
    [+1],
    [-1],
    [-1],
    [+1],
    [-1]]).transpose()

Tag = Tag.flatten()

a = SVM(Original_Data, Tag)

a.train()

for i in range(Original_Data.shape[1]):
    if Tag[i] == +1:
        pyplot.plot(Original_Data[0][i], Original_Data[1][i], "or")
    else:
        pyplot.plot(Original_Data[0][i], Original_Data[1][i], "ob")

x = numpy.arange(-5, +5, 0.01)
pyplot.plot(x, -((a.W[0] * x + a.b)/a.W[1]))


print "Support Vector"
for i in range(Original_Data.shape[1]):
    if a.SupVec[i] == True:
        print Original_Data[0][i], Original_Data[1][i]


pyplot.show()
