"""
Programmer  :   EOF
Date        :   2015.12.17
File        :   semme.py

File Description:
	AdaBoost is a machine learning meta-algorithm. 
That is the short for "Adaptive Boosting".

"""
import numpy
from decisionStump import *

class SAMME:

    def __init__(self, Mat, Label, WeakerClassifier = DecisionStump):
        """
        self._Mat: A matrix which store the samples. Every column 
                   vector in this matrix is a point of sample.
        self._Label: 
    	self.W: A vecter which is the weight of weaker classifier
    	self.N: A number which descripte how many weaker classifier
    		is enough for solution.
	"""
        self._Mat   = numpy.array(Mat)
        self._Label = numpy.array(Label)

        self.SamplesDem = self._Mat.shape[0]
        self.SamplesNum = self._Mat.shape[1]

        # Make sure that the inputed data's demention is right.
        assert self.SamplesNum == self._Label.size

        self.Weaker = WeakerClassifier

        """--------------difference with AdaBoost-----------------"""

        self.labels     = numpy.unique(self._Label)
        self.ClassesNum = len(self.labels)

        self.output = numpy.array(
                        [[0.0 for i in range(self.SamplesNum)] 
                              for j in range(self.ClassesNum)])

        for i in range(self.SamplesNum):
            for j in range(self.ClassesNum):
                if self._Label[i] == self.labels[j]:
                    self.output[j][i] = 1.0
                else:
                    self.output[j][i] = -1.0/(self.ClassesNum - 1)
        """-------------------------------------------------------"""

        # Initialization of weight
        self.W = [1.0/self.SamplesNum for i in range(self.SamplesNum)]

        self.N = 0
        self.G = {}
        self.alpha = {}

    def is_good_enough(self):
        output = numpy.zeros((self.SamplesNum, 1))
        for i in range(self.N+1):
            output += self.G[i].prediction(self._Mat) * self.alpha[i]

        output = numpy.sign(output)
        output = output.flatten()

        if output.tolist() == self._Label.tolist():
            return True
        else:
            return False


    def train(self, M = 4):
	"""
	function @train() is the main process which run 
	AdaBoost algorithm.

	@M : Upper bound weaker classifier. How many weaker 
        classifier will be used to construct a strong 
	classifier.
	"""

        for m in range(M):
            self.G[m] = self.Weaker(self._Mat, self._Label, self.W)
            self.G[m].train()

            p = self.G[m].p

            K = self.ClassesNum

            tmp = 0.0
            for i in range(self.ClassesNum):
                tmp += numpy.log(p[i])

            tmp /= (1.0/self.ClassesNum)

            self.G[m] = {}
            for i in range(self.ClassesNum):
                self.G[m][i] = (K - 1) * (numpy.log(p[i]) - tmp)

            for i in range(self.SamplesNum):
                tmp = 0.0
                for j in range(self.ClassesNum):
                    tmp += self.output[j][i] * numpy.log(p[j])

                self.W[i] *= numpy.exp((-(K - 1)/K) * tmp)

            """
            if self.is_good_enough():
                print (self.N + 1) ," weak classifier is enough to ",\
                      "classify the inputed sample points"
                print "Training Done :)"
                break
            """
            self.N += 1

    def prediction(self, Mat):

        Mat = numpy.array(Mat)
        output = numpy.zeros((Mat.shape[1], 1))
        for i in range(self.N + 1):
            output += self.G[i].prediction(Mat) * self.alpha[i]

        print output
        output = numpy.sign(output)

        return output
