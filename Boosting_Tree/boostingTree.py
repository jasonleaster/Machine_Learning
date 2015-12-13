"""
Programmer  :   EOF
Date        :   2015.11.22
File        :   adaboost.py

"""
import numpy
from decisionStump import *

class BoostingTree:

    def __init__(self, Mat, Exp, WeakerClassifier = DecisionStump):
        """
        self._Mat: A matrix which store the samples. Every column 
                   vector in this matrix is a point of sample.
        self._Exp: The expected val of training samples.
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

        self.N = 0
        self.T = {}

        self.residual = numpy.array(self._Exp)

        self.theta = 0.1

    def is_good_enough(self):
        output = numpy.array([0.0 for i in range(self.SamplesNum)])

        #Use all weak classifier to do prediction and construct into
        #a stronger classifier
        for i in range(self.N + 1):
            output += numpy.array(self.T[i].prediction(self._Mat))
            summer = 0
            for j in range(self.SamplesNum):
                summer += (output[j] - self._Exp[j]) ** 2

        #return Ture, if the variance smaller than tolerance
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
            #We use residual as expected output to train the weak classifier
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
        output = numpy.array([0.0 for i in range(Mat.shape[1])])

        for i in range(self.N + 1):
            output += numpy.array(self.T[i].prediction(Mat))

        return output
