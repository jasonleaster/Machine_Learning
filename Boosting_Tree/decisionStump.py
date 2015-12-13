"""
Programmer  :   EOF
Date        :   2015.11.22
File        :   weakclassifier.py

"""
import numpy

class DecisionStump:

    def __init__(self, Mat, Exp, W = None):
        self._Mat      = numpy.array(Mat)
        self._Exp      = numpy.array(Exp)

        self.SampleDem = self._Mat.shape[0]
        self.SampleNum = self._Mat.shape[1]

        if W == None:
            self.weight = [1.0/self.SampleNum for i in range(self.SampleNum)]
        else:
            self.weight = numpy.array(W)

        self.split_point = numpy.array(
                [[0.0 for i in range(self.SampleNum -1)] \
                    for j in range(self.SampleDem)])

        self.cost = numpy.array(
                [[0.0 for i in range(self.SampleNum -1)] \
                    for j in range(self.SampleDem)])

        self.threshold = [0.0 for i in range(self.SampleDem)]
        self.output = [0.0, 0.0]

    def _sort(self, depth):
        dem = depth % self.SampleDem

        for i in range(self.SampleNum):
            for j in range(i+1, self.SampleNum):
                if self._Mat[dem][i] > self._Mat[dem][j]:
                    #swap sample with index @i and  @j
                    self._Mat[:, i], self._Mat[:, j] = \
                        numpy.array(self._Mat[:, j]),\
                        numpy.array(self._Mat[:, i])

                    self._Exp[i], self._Exp[j] = \
                        numpy.array(self._Exp[j]),\
                        numpy.array(self._Exp[i])

    def _split_point(self, depth):
        self._sort(depth)

        for i in range(self.SampleNum-1):
            self.split_point[depth][i] = 0.5* (self._Mat[depth][i] + self._Mat[depth][i+1])


    def _mean(self, depth, threshold):
        c1 = 0
        c2 = 0
        N1 = 0
        N2 = 0
        for i in range(self.SampleNum):
            p = self._Mat[depth][i]
            if p < threshold:
                c1 += self._Exp[i]
                N1 += 1
            else:
                c2 += self._Exp[i]
                N2 += 1

        c1 /= N1
        c2 /= N2

        return (c1, c2)

    def CalCost(self, depth):
        self._split_point(depth)
        for i in range(len(self.cost[depth])):
            (c1, c2) = self._mean(depth, self.split_point[depth][i])
            summer = 0
            for j in range(self.SampleNum):
                p = self._Mat[depth][j]
                if p < self.split_point[depth][i]: # Left Tree
                    summer += (self._Exp[j] - c1)**2
                else: # Right Tree
                    summer += (self._Exp[j] - c2)**2

            self.cost[depth][i] = summer

    def findThreshold(self, depth):
        minVal = numpy.inf
        minCostIndex = 0
        for i in range(len(self.cost[depth])):
            if minVal > self.cost[depth][i]:
                minVal = self.cost[depth][i]
                minCostIndex = i

        return self.split_point[depth][minCostIndex]

    def train(self):
        """
            function @train() will return the mininum error value
        in the procedure of training.

        W is a N length Vector
        """
        for demention in range(self.SampleDem):
            self.CalCost(demention)
            self.threshold[demention] = self.findThreshold(demention)
            self.output = self._mean(demention, self.threshold[demention])

    def prediction(self, Mat):
        Mat = numpy.array(Mat)
        out = [None for i in range(Mat.shape[1])]
        for i in range(Mat.shape[1]):
            for j in range(Mat.shape[0]):
                if Mat[j,i] < self.threshold[j]:
                    out[i] = self.output[0] 
                else:
                    out[i] = self.output[1] 

        return out
