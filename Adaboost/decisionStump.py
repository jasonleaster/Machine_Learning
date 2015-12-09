"""
Programmer  :   EOF
Date        :   2015.11.22
File        :   weakclassifier.py

"""
import numpy

class DecisionStump:

    def __init__(self, Mat, Tag, W = None):
        self._Mat      = numpy.array(Mat)
        self._Tag      = numpy.array(Tag)
        self.labels    = numpy.unique(Tag)

        self.SampleDem = self._Mat.shape[0]
        self.SampleNum = self._Mat.shape[1]

        if W == None:
            self.weight = [1.0/self.SampleNum for i in range(self.SampleNum)]
        else:
            self.weight = numpy.array(W)

    def prediction(self, Mat):
        th    = self.opt_threshold
        label = self.opt_label
        index = self.opt_demention

        output = numpy.ones((Mat.shape[1], 1))
        output[Mat[index, :] * label < th * label] = -1

        return output


    def optimal(self, i, label, steps):
        buttom = numpy.min(self._Mat[i, :])
        up     = numpy.max(self._Mat[i, :]) 

        minErr = 1.0
        accTh  = 0

        accuracy = (up - buttom)/steps

        for t in numpy.arange(buttom, up, accuracy):

            output = numpy.ones((self.SampleNum, 1))

            output[self._Mat[i, :] * label < t * label] = -1

            errorRate = 0.0
            for index in range(self.SampleNum):
                if output[index] != self._Tag[index]:
                    errorRate += self.weight[index]

            if errorRate < minErr:
                minErr = errorRate
                accTh  = t

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
