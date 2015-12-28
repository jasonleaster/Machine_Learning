"""
Programmer  :   EOF
Date        :   2015.12.28
File        :   tester6.py

File Description:
    This is a test script which get two huandreds of sample
points from sklearn.datasets

This test prove out that my AdaBoost Algorithm is correct :)

Just Enjoy it.
"""

import numpy
import matplotlib.pyplot as pyplot
from adaboost import AdaBoost
from sklearn import datasets

"""
Samples for AdaBoost
"""
Original_Data, Tag = datasets.make_hastie_10_2(n_samples    = 200, 
                                              random_state  = 1)
Original_Data = Original_Data.transpose()


for i in range(len(Tag)):
    if Tag[i] == 1:
        pyplot.plot(Original_Data[0][i], Original_Data[1][i], \
                    '+r', markersize = 10)
    else:
        pyplot.plot(Original_Data[0][i], Original_Data[1][i], \
                    '+b', markersize = 10)
pyplot.title("Sample Points")
pyplot.show()

a = AdaBoost(Original_Data, Tag)

a.train(10000)
