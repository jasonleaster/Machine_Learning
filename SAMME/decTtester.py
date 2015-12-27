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
[0],
[1],
[2],
[3],
[4],
[5],
[6],
[7],
[8],
[9]
]).transpose()

Tag = numpy.array([
["+2"],
["+2"],
["+2"],
["-1"],
["-1"],
["-1"],
["+1"],
["+1"],
["+1"],
["-1"],
]).transpose()

Tag = Tag.flatten()

discrete = [False]

a = SAMME(Original_Data, Tag, Discrete = discrete)

a.train()

a.prediction(a._Mat)

