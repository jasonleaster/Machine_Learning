import numpy
from adaboost import AdaBoost

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
[+1],
[+1],
[+1],
[-1],
[-1],
[-1],
[+1],
[+1],
[+1],
[-1],
]).transpose()

Tag = Tag.flatten()

a = AdaBoost(Original_Data, Tag)

a.train(5)

