"""
Programmer  :   EOF
Cooperator  :   Wei Chen.
Date        :   2015.11.22
File        :   adaboost.py

File Description:
	AdaBoost is a machine learning meta-algorithm. 
That is the short for "Adaptive Boosting".

Thanks Wei Chen. Without him, I can't understand AdaBoost in this short time. We help each other and learn this algorithm.

"""
import numpy
from decisionStump import *
import matplotlib.pyplot as pyplot

class AdaBoost:

    def __init__(self, Mat, Tag, WeakerClassifier = DecisionStump):
        """
        self._Mat: A matrix which store the samples. Every column 
                   vector in this matrix is a point of sample.
        self._Tag: 
    	self.W: A vecter which is the weight of weaker classifier
    	self.N: A number which descripte how many weaker classifier
    		is enough for solution.
	"""
        self._Mat = numpy.array(Mat) * 1.0
        self._Tag = numpy.array(Tag) * 1.0

        self.SamplesDem = self._Mat.shape[0]
        self.SamplesNum = self._Mat.shape[1]

        # Make sure that the inputed data's demention is right.
        assert self.SamplesNum == self._Tag.size

        self.Weaker = WeakerClassifier

        # Initialization of weight
        self.W = [1.0/self.SamplesNum for i in range(self.SamplesNum)]

        self.N = 0
        self.G = {}
        self.alpha = {}

        self.accuracy = []

    def is_good_enough(self):
        output = numpy.zeros((self.SamplesNum, 1))
        for i in range(self.N+1):
            output += self.G[i].prediction(self._Mat) * self.alpha[i]

        output = numpy.sign(output)
        output = output.flatten()

        e = numpy.count_nonzero(output ==self._Tag)/(self.SamplesNum*1.) 
        self.accuracy.append( e )

        if output.tolist() == self._Tag.tolist():
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
            self.G[m] = self.Weaker(self._Mat, self._Tag, self.W)
            self.G[m].train()

            output = self.G[m].prediction(self._Mat)

            errorRate = self.G[m].opt_errorRate

            self.alpha[m] = 0.5 * numpy.log((1-errorRate)/errorRate)
            
            if self.is_good_enough():
                print (self.N + 1) ," weak classifier is enough to ",\
                      "classify the inputed sample points"
                print "Training Done :)"
                break

            Z = 0.0
            for i in range(self.SamplesNum):
                Z += self.W[i] * numpy.exp(-self.alpha[m] * self._Tag[i] * output[i])

            for i in range(self.SamplesNum):
                self.W[i] = (self.W[i] / Z) * numpy.exp(-self.alpha[m] * self._Tag[i] * output[i])

            self.N += 1

            if self.accuracy[self.N-1] > 0.97 and self.N > 50:
                self.showErrRates()
                return

    def prediction(self, Mat):

        Mat = numpy.array(Mat)
        output = numpy.zeros((Mat.shape[1], 1))
        for i in range(self.N + 1):
            output += self.G[i].prediction(Mat) * self.alpha[i]

        print output
        output = numpy.sign(output)

        return output

    def showErrRates(self):
        pyplot.title("The changes of accuracy (Figure by Jason Leaster)")
        pyplot.xlabel("Iteration times")
        pyplot.ylabel("Accuracy of Prediction")
        pyplot.plot([i for i in range(self.N)], self.accuracy, '-.', label = "Accuracy * 100%")

        pyplot.show()
