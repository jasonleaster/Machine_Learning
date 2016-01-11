"""
Programmer  :   EOF
File        :   tester2.py
Date        :   2016.01.10
E-mail      :   jasonleaster@163.com

Description :

"""
import numpy
from matplotlib import pyplot
from km import KMeans

Original_Data = numpy.array([
    [1, 1.5],
    [1, 0.5],
    [0.5, 0.5],
    [1.5, 1.5],
    [5, 5],
    [6, 5.5],
    [4, 5],
    [5, 1],
    [6, 0.5],
    [7, 1.5],
    [1, 10],
    [1.5, 11]
    ]).transpose()


a = KMeans(Original_Data, K = 4)

for i in range(a.SampleNum):
    pyplot.plot(Original_Data[0][i], Original_Data[1][i], "+r", markersize=12)


pyplot.title("Original Training Data (Figure by Jason Leaster)")
pyplot.axis([-2, 14, -2, 14])
pyplot.show()

a.train()

a.show()
