"""
Programmer  :   EOF
File        :   tester.py
Date        :   2015.12.29
E-mail      :   jasonleaster@163.com

Description :

"""
import numpy
from matplotlib import pyplot
from km import KMeans

Original_Data = numpy.array([
    [1, 5],
    [1, 6],
    [1, 7],
    [2, 1],
    [2, 2],
    [3, 1],
    [4, 1],
    [5, 2],
    [5, 3],
    [6, 5],
    [6, 6],
    [6, 7]
    ]).transpose()


a = KMeans(Original_Data, K = 3)

for i in range(a.SampleNum):
    pyplot.plot(Original_Data[0][i], Original_Data[1][i], "+r", markersize=12)


pyplot.title("Original Training Data (Figure by Jason Leaster)")
pyplot.axis([-2, 10, -2, 10])
pyplot.show()

a.train()

a.show()
