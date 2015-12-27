"""
Programmer  :   EOF
File        :   tester.py
Date        :   2015.12.10
E-mail      :   jasonleaster@163.com

    All test data in this test module come from chapter2 in
    <<The methods of statistic >> by Hang-Li.
"""
import numpy
import matplotlib.pyplot as plt
from perceptron import Perceptron

Original_Date = numpy.array([
    [3,3],
    [4,3],
    [1,1]
])

Tag = numpy.array([
    [+1],
    [+1],
    [-1],
    ])

# Create a perceptron
a = Perceptron(Original_Date, Tag)

# training it.
a.train()

# ---------- draw originall data -----------
x = numpy.arange(-1, 10, 0.01)
plt.plot(x, -(a.w[1] * x + a.b)/a.w[0])

for i in range(len(Tag)):
    if Tag[i] == +1:
        plt.plot(Original_Date[i][0], Original_Date[i][1], 'or')
    else:
        plt.plot(Original_Date[i][0], Original_Date[i][1], 'ob')
# ---------------  end --------------------

UnknowPoints = numpy.array([
[+5,+5],
[-1,-1]]).transpose()

# Prediction. Try to classify unkonw points
output = a.prediction(UnknowPoints)

for i in range(UnknowPoints.shape[1]):
    if output[i] == +1:
        plt.plot(UnknowPoints[0][i], \
                 UnknowPoints[1][i], '+r', markersize = 20)
    else:
        plt.plot(UnknowPoints[0][i], \
                 UnknowPoints[1][i], '+b', markersize = 20)

plt.show()
