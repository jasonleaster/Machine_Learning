"""
Programmer  :   EOF
File        :   km.py
Date        :   2015.12.29
E-mail      :   jasonleaster@163.com

Description :
"""

import numpy
from numpy.random import uniform as rand


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

        self.scope     = [[min(self._Mat[i, :]), max(self._Mat[i, :])] for i in range(self.SampleDem)]
        """
        Initialization of @meanVals in randomly in the scope.
        """
        self.meanVal   = numpy.array([[rand(self.scope[i][0], self.scope[i][1])
                            for i in range(self.SampleDem)] 
                            for j in range(self.classNum)]).transpose()

        self.classification = [(None, None) for i in range(self.SampleNum)]

    def train(self):
        while True:
            for i in range(self.SampleNum):
                minDis = +numpy.inf
                label  = None
                for k in range(self.K):
                    d = self.distance(self._Mat[:, i], selsf.meanVal[:, k])
                    if d < minDis:
                        minDis = d
                        label  = k
                self.classification[i][0] = label
                self.classification[i][1] = minDis

            for k in range(self.classNum):
                opt_point = None
                minDis = +numpy.inf
                for i in range(self.SampleNum):
                    if self.classification[i][0] == k and\
                       self.classification[i][1] < minDis:
                        minDis    = self.classification[i][1]
                        opt_point = self._Mat[:, i]

                self.meanVal[k] = opt_point

            if stopOrNot(self):
                return

    def minDisInClass(self, k):
    
        assert k >= 0

        dis = []
        for i in range(self.SampleNum):
            if self.classification[i][0] == k:
                summer = 0.
                for j in range(self.SampleNum):
                    if self.classification[j][0] = k:
                        summer += self.distance(self._Mat[:, i], self._Mat[:, j])

                dis.append(summer)

        return min(dis)

    def stopOrNot(self):
        for k in range(self.classNum):
            summer = 0.
            for i in range(self.SampleNum):
                if self.classification[i][0] == k:
                    summer += self.distance(self.meanVal(k), self._Mat[:, i])
    
            if summer >= self.minDisInClass(k):
                return False


        return True

