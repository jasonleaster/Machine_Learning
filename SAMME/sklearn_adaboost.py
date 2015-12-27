from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
import numpy

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
    ])

Tag = numpy.array([
    [+2],
    [+2],
    [+2],
    [-1],
    [-1],
    [-1],
    [+1],
    [+1],
    [+1],
    [-1],
    ]).transpose()

Tag = Tag.flatten()

a = AdaBoostClassifier(DecisionTreeClassifier(max_depth=2), n_estimators = 600, learning_rate = 1)

a.fit(Original_Data, Tag)

print a.predict(Original_Data)
