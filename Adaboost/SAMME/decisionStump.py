"""
Programmer  :   EOF
Date        :   2015.11.22
File        :   weakclassifier.py

"""
import numpy

class DecisionStump:

    def __init__(self, Mat, Label, W = None):
        self._Mat      = numpy.array(Mat)
        self._Label    = numpy.array(Label)
        self.labels    = numpy.unique(Label)

        self.SamplesDem = self._Mat.shape[0]
        self.SamplesNum = self._Mat.shape[1]

        assert W != None

        self.weight = numpy.array(W)

        # difference
        self.ClassesNum = len(self.labels)
        self.p = [0.0 for i in range(self.ClassesNum)]


    def train(self):
        for i in range(self.ClassesNum):
            for j in range(self.SamplesNum):
                if self._Label[j] == self.labels[i]:
                    self.p[i] += self.weight[j]
                    
