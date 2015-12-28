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
from samme import SAMME

Original_Data = numpy.array([
    ['teenager',        'no',   'no',   0.0],
    ['teenager',        'no',   'no',   1.0],
    ['teenager',        'yes',  'no',   1.0],
    ['teenager',        'yes',  'yes',  0.0],
    ['teenager',        'no',   'no',   0.0],
    ['senior citizen',  'no',   'no',   0.0],
    ['senior citizen',  'no',   'no',   1.0],
    ['senior citizen',  'yes',  'yes',  1.0],
    ['senior citizen',  'no',   'yes',  2.0],
    ['senior citizen',  'no',   'yes',  2.0],
    ['old pepple',      'no',   'yes',  2.0],
    ['old pepple',      'no',   'yes',  1.0],
    ['old pepple',      'yes',  'no',   1.0],
    ['old pepple',      'yes',  'no',   2.0],
    ['old pepple',      'no',   'no',   0.0],
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

Tag = Tag.flatten()

discrete = [ True for i in range(Original_Data.shape[0])]
 
a = SAMME(Original_Data, Tag, discrete)

a.train()

print a.prediction(a._Mat)
