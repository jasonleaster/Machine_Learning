"""
Programmer  :   EOF
File        :   tester.py
Date        :   2015.12.29
E-mail      :   jasonleaster@163.com

Description :

"""
import numpy
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

a.train()
