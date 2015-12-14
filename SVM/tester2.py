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
[10.75  , 14.95],
[11.35  , 14.05],
[12     , 13.65],
[12.75  , 13.75],
[14.2   , 14.85],
[13.2   , 15.8],
[12.1   , 16.55],
[11.3   , 15.8],
[12.7   , 14.7],
[12.8   , 14.55],
[12.35  , 15.75],
[8.55   , 17.2],
[8.15   , 15.4],
[8.7    , 13.95],
[9.75   , 12.25],
[11.9   , 11.1],
[14.45  , 11.3],
[16.7   , 12.9],
[16.9   , 14.45],
[16.4   , 15.95],
[15     , 17.7],
[12.2   , 18.8],
[9.85   , 18.45],
[8.2    , 17.5],
[7.4    , 15.85],
[7.8    , 13.6],
[9.6    , 12.1],
[12.3   , 11.2],
[14.95  , 12.2],
[15.2   , 13.05],
[15.55  , 13.65],
[15.9   , 14.6],
[15.5   , 16.45],
[14.75  , 17.9],
[13.1   , 18.5],
[10.8   , 19.4],
[11.05  , 18.7]]).transpose()


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
[+1],
[+1],
[+1],
[+1]]).transpose()

Tag = Tag.flatten()

#a = SVM(Original_Data, Tag)

#a.train()

for i in range(Original_Data.shape[1]):
    if Tag[i] == +1:
        pyplot.plot(Original_Data[0][i], Original_Data[1][i], 'or')
    else:
        pyplot.plot(Original_Data[0][i], Original_Data[1][i], 'ob')


#x = numpy.arange(-5, +5, 0.01)
#pyplot.plot(x, -((a.W[1] * x + a.b)/a.W[0]))

pyplot.show()
