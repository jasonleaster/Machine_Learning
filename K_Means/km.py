"""
Programmer  :   EOF
File        :   km.py
Date        :   2015.12.29
E-mail      :   jasonleaster@163.com

Description :
    The implementation of K-Means Model.

"""

import numpy
from numpy.random import uniform as rand
from matplotlib import pyplot

def euclidean_distance(list1, list2):
    assert isinstance(list1, list)
    assert isinstance(list2, list)

    summer = 0
    for item1, item2 in zip(list1, list2):
        summer += (item1 - item2)**2
    return numpy.sqrt(summer)


class KMeans:
    def __init__(self, Mat, K, disFunc = euclidean_distance):
        self._Mat   = numpy.array(Mat)
        
        self.SampleDem = self._Mat.shape[0]
        self.SampleNum = self._Mat.shape[1]

        self.classNum  = K
        self.distance  = disFunc

        self.scope     = [[min(self._Mat[i, :]), 
                           max(self._Mat[i, :])] 
                           for i in range(self.SampleDem)]
        """
        Initialization of @meanVals in randomly in the scope.
        """
        self.meanVal   = numpy.array([[rand(self.scope[i][0], self.scope[i][1])
                            for i in range(self.SampleDem)] 
                            for j in range(self.classNum)]).transpose()

        self.classification = [[None, None] for i in range(self.SampleNum)]

        for i in range(self.SampleNum):
            minDis = +numpy.inf
            label  = None
            for k in range(self.classNum):
                d = self.distance(self._Mat[:, i].tolist(), self.meanVal[:, k].tolist())

                if d < minDis:
                    minDis = d
                    label  = k

            self.classification[i][0] = label
            self.classification[i][1] = minDis

    def train(self):
        while True:

            if self.stopOrNot():
                return

            for k in range(self.classNum):
                (minDis, self.meanVal[:, k]) = self.minDisInClass(k)


    def minDisInClass(self, k):
    
        assert k >= 0

        minDis = +numpy.inf
        for i in range(self.SampleNum):
            if self.classification[i][0] == k:
                summer = 0.
                for j in range(self.SampleNum):
                    if self.classification[j][0] == k:
                        summer += self.distance(self._Mat[:, i].tolist(), 
                                                self._Mat[:, j].tolist())

                if minDis > summer:
                    minDis = summer
                    opt_point = self._Mat[:, i]

        return (minDis, opt_point)

    def stopOrNot(self):
        for k in range(self.classNum):
            summer = 0.
            for i in range(self.SampleNum):
                if self.classification[i][0] == k:
                    summer += self.distance(self.meanVal[:,k].tolist(), self._Mat[:, i].tolist())
    
            (minDis, _) = self.minDisInClass(k)
            if summer > minDis:
                return False

        return True

    def show(self):
        """
        Only support two demention feature samples!
        Just a toy function.
        """
        assert self.SampleDem == 2

        print "Means: ", self.meanVal
        width = 2

        for k in range(self.classNum):
            for i in range(self.SampleNum):
                if self.classification[i][0] == 0:
                    pyplot.plot(self._Mat[0][i], self._Mat[1][i], "or")
                elif self.classification[i][0] == 1:
                    pyplot.plot(self._Mat[0][i], self._Mat[1][i], "og")
                elif self.classification[i][0] == 2:
                    pyplot.plot(self._Mat[0][i], self._Mat[1][i], "ob")

        pyplot.axis([int(self.scope[0][0]) - width, int(self.scope[0][1]) + width, 
                    int(self.scope[1][0]) - width, int(self.scope[1][1]) + width])
        pyplot.show()
