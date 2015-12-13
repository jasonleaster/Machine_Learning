"""
Programmer  :   EOF
Date        :   2015.11.22
File        :   adaboost.py

File Description:
    Boosting Tree
"""
import numpy
from decisionStump import *

class BoostingTree:

    def __init__(self, Mat, Exp, WeakerClassifier = DecisionStump):
        """
        self._Mat: A matrix which store the samples. Every column 
                   vector in this matrix is a point of sample.
        self._Tag: 
    	self.W: A vecter which is the weight of weaker classifier
    	self.N: A number which descripte how many weaker classifier
    		is enough for solution.
	"""
        self._Mat = numpy.array(Mat) * 1.0
        self._Exp = numpy.array(Exp) * 1.0

        self.SamplesDem = self._Mat.shape[0]
        self.SamplesNum = self._Mat.shape[1]

        # Make sure that the inputed data's demention is right.
        assert self.SamplesNum == self._Exp.size

        self.Weaker = WeakerClassifier

        # Initialization of weight
        self.W = [1.0/self.SamplesNum for i in range(self.SamplesNum)]

        self.N = 0
        self.T = {}

        self.residual = numpy.array(self._Exp)

        self.theta = 0.2

    def is_good_enough(self):
        output = numpy.array([0.0 for i in range(self.SamplesNum)])

        for i in range(self.N + 1):
            output += numpy.array(self.T[i].prediction(self._Mat))
            summer = 0
            for j in range(self.SamplesNum):
                summer += (output[j] - self._Exp[j]) ** 2

        if summer < self.theta:
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
            self.T[m] = self.Weaker(self._Mat, self.residual)
            self.T[m].train()

            output = self.T[m].prediction(self._Mat)

            for i in range(len(output)):
                self.residual[i] -= output[i]

            if self.is_good_enough():
                print (self.N + 1) ," weak classifier is enough to ",\
                      "classify the inputed sample points"
                print "Training Done :)"
                break

            self.N += 1

    def prediction(self, Mat):

        Mat = numpy.array(Mat)
        output = numpy.zeros((Mat.shape[1], 1))
