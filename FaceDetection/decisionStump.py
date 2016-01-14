"""
Programmer  :   EOF
Date        :   2015.11.22
File        :   weakclassifier.py

"""
import numpy
from matplotlib import pyplot as plt
from config import *

class DecisionStump:

    def __init__(self, Mat = None, Tag = None, W = None, train = True):
        if train == True:
            self._Mat = numpy.array(Mat)
            self._Tag = numpy.array(Tag)

            self.labels  = numpy.unique(Tag)

            self.SampleDem = self._Mat.shape[0]
            self.SampleNum = self._Mat.shape[1]

            if W == None:
                self.weight = [1.0/self.SampleNum for i in range(self.SampleNum)]
            else:
                self.weight = numpy.array(W)

            # For acceleration, sort the original sample by index but not the sample direactly.

            self.indexes = [i for i in range(self.SampleNum)]

    def constructor(self, demention, label, threshold):

        self.opt_demention = demention
        self.opt_threshold = threshold
        self.opt_label     = label

        return self

    def __str__(self):

        string  = "opt_threshold:" + str(self.opt_threshold) + "\n"
        string += "opt_demention:" + str(self.opt_demention) + "\n"
        string += "opt_errorRate:" + str(self.opt_errorRate) + "\n"
        string += "opt_label    :" + str(self.opt_label    ) + "\n"
        string += "weights      :" + str(self.weight)        + "\n"

        return string

    def __sort__(self, d):

        idx = self.indexes
        for i in range(len(idx)):
            for j in range(i+1, len(idx)):
                if self._Mat[d, idx[i] ] > self._Mat[d, idx[j] ]:
                    idx[i], idx[j] = idx[j], idx[i]

    def prediction(self, Mat):

        #get a copy
        Mat = numpy.array(Mat)

        th    = self.opt_threshold
        label = self.opt_label
        index = self.opt_demention

        output = numpy.array([1 for i in range(Mat.shape[1])])

        output[Mat[index, :] * label < th * label] = -1

        return output

    def optimal(self, d, label, steps):

        buttom = numpy.min(self._Mat[d, :])
        up     = numpy.max(self._Mat[d, :]) 

        self.__sort__(d)

        idx = self.indexes

        minCost = 1.0
        accTh   = 0.

        accuracy = (up - buttom)/steps

        for t in numpy.arange(buttom, up, accuracy):

            output = numpy.ones((self.SampleNum, 1))

            for i in range(self.SampleNum):
                if self._Mat[d, idx[i] ] * label < t * label:
                    output[ idx[i] ] = -1
                
            cost = 0.0
            for i in range(self.SampleNum):
                if output[ idx[i] ] != self._Tag[ idx[i] ]:
                    cost += self.weight[ idx[i] ]

            if cost < minCost:
                minCost = cost
                accTh  = t    #Acceptable threshold

        return minCost, accTh

    def train(self, steps = 100):

        self.opt_errorRate = 1.0

        processing = 0.
        for demention in range(self.SampleDem):

            if demention % (self.SampleDem / 10) == 0:
                print "In weaker processing :", processing
                processing += 10.

            for label in self.labels:
                cost, th = self.optimal(demention, label, steps)

                if cost < self.opt_errorRate:
                    self.opt_cost = cost
                    self.opt_demention = demention
                    self.opt_threshold = th
                    self.opt_label     = label

        output = self.prediction(self._Mat)

        errorRate = numpy.count_nonzero(output == self._Tag) * 1. / self.SampleNum

        return errorRate


    def show(self, d):
        self.__sort__(d)

        idx = self.indexes

        for i in range(self.SampleNum):
            if self._Tag[idx[i]] == 1:
                plt.plot(idx[i], self._Mat[d, idx[i]], 'or')
            else:
                plt.plot(idx[i], self._Mat[d, idx[i]], 'ob')

        plt.show()


