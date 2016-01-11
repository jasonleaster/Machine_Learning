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
from config import *
import matplotlib.pyplot as pyplot

class AdaBoost:

    def __init__(self, Mat = None, Tag = None, WeakerClassifier = DecisionStump, train = True):
        """
        self._Mat: A matrix which store the samples. Every column 
                   vector in this matrix is a point of sample.
        self._Tag: 
    	self.W: A vecter which is the weight of weaker classifier
    	self.N: A number which descripte how many weaker classifier
    		is enough for solution.
	"""
        if train == True:
            self._Mat = numpy.array(Mat) * 1.0
            self._Tag = numpy.array(Tag) * 1.0

            self.SamplesDem = self._Mat.shape[0]
            self.SamplesNum = self._Mat.shape[1]

            # Make sure that the inputed data's demention is right.
            assert self.SamplesNum == self._Tag.size


            # Initialization of weight
            pos_W = [1.0/(2 * POSITIVE_SAMPLE) for i in range(POSITIVE_SAMPLE)]

            neg_W = [1.0/(2 * NEGATIVE_SAMPLE) for i in range(NEGATIVE_SAMPLE)]
            self.W = pos_W + neg_W

            self.accuracy = []

        self.Weaker = WeakerClassifier

        self.G = {}
        self.alpha = {}
        self.N = 0
        self.detectionRate = 0.


    def is_good_enough(self):
        output = self.prediction(self._Mat)

        e = numpy.count_nonzero(output ==self._Tag)/(self.SamplesNum*1.) 
        self.accuracy.append( e )

        self.detectionRate = numpy.count_nonzero(output[0:POSITIVE_SAMPLE] == 1) * 1./ POSITIVE_SAMPLE

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

            errorRate = self.G[m].opt_errorRate

            self.alpha[m] = 0.5 * numpy.log((1-errorRate)/errorRate)

            output = self.G[m].prediction(self._Mat)
            
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

            print "errorRate:", errorRate
            print "Accuracy:", self.accuracy[-1]
            print "DetectionRate:", self.detectionRate

            if self.accuracy[self.N-1] > 0.90 or self.detectionRate > 0.99:
                self.showErrRates()
                return

        print "can not meet the request! :("

    def prediction(self, Mat):

        Mat = numpy.array(Mat)

        output = numpy.array([ 0. for i in range(Mat.shape[1])])

        for i in range(self.N):
            output += self.G[i].prediction(Mat) * self.alpha[i]

        for i in range(len(output)):
            if output[i] > 0.:
                output[i] = +1
            else:
                output[i] = -1

        return output

    def showErrRates(self):
        pyplot.title("The changes of accuracy (Figure by Jason Leaster)")
        pyplot.xlabel("Iteration times")
        pyplot.ylabel("Accuracy of Prediction")
        pyplot.plot([i for i in range(self.N)], self.accuracy, '-.', label = "Accuracy * 100%")

        pyplot.show()

    def constructor(self, alpha, weakers):
        pass
