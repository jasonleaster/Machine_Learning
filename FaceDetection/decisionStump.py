"""
Programmer  :   EOF
Date        :   2015.11.22
File        :   weakclassifier.py

"""
import numpy
from config import *

def sortByDem(Mat, d):
    assert isinstance(Mat, numpy.array)

    SampleDem = Mat.shape[0]
    SampleNum = Mat.shape[1]

    for i in range(SampleNum):
        for j in range(i+1, SampleNum):
            if Mat[d, i] > Mat[d, j]:
                Mat[:, i], Mat[:, j] = \
                numpy.array(Mat[:, j]),
                numpy.array(Mat[:, i])

    return Mat

class DecisionStump:

    def __init__(self, Mat = None, Tag = None, W = None, train = True):
        if train == True:
            self._Mat = numpy.array(Mat)
            self._Tag = numpy.array(Tag)

            self._oriMat = numpy.array(Mat)
            self._oriTag = numpy.array(Tag)

            self.labels  = numpy.unique(Tag)

            self.SampleDem = self._Mat.shape[0]
            self.SampleNum = self._Mat.shape[1]

            if W == None:
                self.weight = [1.0/self.SampleNum for i in range(self.SampleNum)]
            else:
                self.weight = numpy.array(W)

    def constructor(self, demention, label, threshold):

        self.opt_demention = demention
        self.opt_threshold = label
        self.opt_label     = threshold

        return self

    def __str__(self):

        string  = "opt_threshold:" + str(self.opt_threshold) + "\n"
        string += "opt_demention:" + str(self.opt_demention) + "\n"
        string += "opt_errorRate:" + str(self.opt_errorRate) + "\n"
        string += "opt_label    :" + str(self.opt_label    ) + "\n"
        string += "weights      :" + str(self.weight)        + "\n"

        return string

    def __sort__(self, d):

        for i in range(self.SampleNum):
            for j in range(i+1, self.SampleNum):
                if self._Mat[d, i] > self._Mat[d, j]:
                    self._Mat[:, i], self._Mat[:, j] =\
                    numpy.array(self._Mat[:, j]), \
                   numpy.array(self._Mat[:, i])

                    self._Tag[i], self._Tag[j] =\
                    numpy.array(self._Tag[j]), \
                    numpy.array(self._Tag[i])

                    self.weight[i], self.weight[j] =\
                    numpy.array(self.weight[j]),\
                    numpy.array(self.weight[i])


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

        minErr = 1.0
        accTh  = 0

        accuracy = (up - buttom)/steps

        for t in numpy.arange(buttom, up, accuracy):

            output = numpy.ones((self.SampleNum, 1))

            output[self._Mat[d, :] * label < t * label] = -1

            S_plus = 0.
            S_nega = 0.

            for i in range(self.SampleNum):
                if self._Mat[d, i] < t:
                    if self._Tag[i] == 1:
                        S_plus += self.weight[i]
                    else:
                        S_nega += self.weight[i]

                else:
                    break

            errorRate = min(S_plus + (0.5 - S_nega), S_nega + (0.5 - S_plus))
            if errorRate < 0:
                print  "ERROR!!!"

            """
            errorRate = 0.0
            for index in range(self.SampleNum):
                if output[index] != self._Tag[index]:
                    errorRate += self.weight[index]
            """

            if errorRate < minErr:
                minErr = errorRate
                accTh  = t 

        """
        output = numpy.array([1 for i in range(self.SampleNum)])

        output[self._oriMat[d, :] * label < accTh * label] = -1
        
        errorRate = 1.* numpy.count_nonzero(output != self._oriTag)/ self.SampleNum
        """
        return minErr, accTh

    def train(self, steps = 100):
        """
            function @train() will return the mininum error value
        in the procedure of training.

        W is a N length Vector
        """
        self.opt_errorRate = 1.0

        for demention in range(self.SampleDem):
            for label in self.labels:
                err, th = self.optimal(demention, label, steps)

                if err < self.opt_errorRate:
                    self.opt_errorRate = err
                    self.opt_demention = demention
                    self.opt_threshold = th
                    self.opt_label     = label
