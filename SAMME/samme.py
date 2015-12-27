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

    def __init__(self, Mat, Label, WeakerClassifier = DecisionTree):
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

        # Initialization of weight
        self.W = [1.0/self.SamplesNum for i in range(self.SamplesNum)]

        self.N = 0
        self.H = {}
        self.alpha = {}

    def is_good_enough(self):
        pass

    def train(self, M = 4):
	"""
	function @train() is the main process which run 
	AdaBoost algorithm.

	@M : Upper bound weaker classifier. How many weaker 
        classifier will be used to construct a strong 
	classifier.
	"""

        for m in range(M):
            self.H[m] = self.Weaker(self._Mat, self._Label, self.W)

            self.N += 1

    def prediction(self, Mat):

        return output
