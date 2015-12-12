"""
Programmer :    EOF
E-mail     :    jasonleaster@163.com
File       :    naive_bayesian_test.py
Date       :    2015.12.07

"""
from naive_bayesian import *

"""
Test Case 1:
    Inputed data is discrete variable which means that you can classify
    the inputed data in each feature into a collection.

    like this:
        the second feature of data below there is belong to 
        {'S', 'M', 'L'}
"""
Original_Date = numpy.array([
[1, 'S'],
[1, 'M'],
[1, 'M'],
[1, 'S'],
[1, 'S'],
[2, 'S'],
[2, 'M'],
[2, 'M'],
[2, 'L'],
[2, 'L'],
[3, 'L'],
[3, 'M'],
[3, 'M'],
[3, 'L'],
[3, 'L'],
]).transpose()


Tag = numpy.array([
[-1],
[-1],
[+1],
[+1],
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
[-1],
]).transpose()

a = NaiveBayesian(Original_Date, Tag)

a.train()

UnkownPoint = numpy.array([
[2, 'S'],
[2, 'L']
]).transpose()

print "Test Case 1:"
print a.prediction(UnkownPoint)
#-------------------End of test case 1  -------------------------



"""
Test Case 2 which have continuous varibable.
This data come from Wikipedia
"""
Original_Date = numpy.array([
#height weight foot-size
[6.0,   180,    12],
[5.92,  190,    11],
[5.58,  170,    12],
[5.92,  165,    10],
[5.0,   100,    6],
[5.5,   150,    8],
[5.42,  130,    7],
[5.75,  150,    9],
]).transpose()


Tag = numpy.array([
['male'],
['male'],
['male'],
['male'],
['female'],
['female'],
['female'],
['female'],
])

Discrete = [False, False, False]
a = NaiveBayesian(Original_Date, Tag, Discrete)

a.train()

UnkownPoint = numpy.array([
    [6, 130, 8]
]).transpose()
print "Test Case 2:"
print "This people is a ", a.prediction(UnkownPoint)
