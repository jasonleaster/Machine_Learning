"""
Programmer  :   EOF
Date        :   2015.12.17
File        :   semme.py

File Description:
	AdaBoost is a machine learning meta-algorithm. 
That is the short for "Adaptive Boosting". But AdaBoost is
only deal with two-class classification.

SAMME is a variant version of AdaBoost which could be used 
to do multi-class classification.

"""
import numpy

from decisionTree import DecisionTree

def toHashableVal(Vec):
    val = ""
    for i in range(len(Vec)):
        val += str(Vec[i])

    return val


class SAMME:

    def __init__(self, Mat, Label, Discrete = None, WeakerClassifier = DecisionTree):
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

        self.labels = numpy.unique(self._Label)
        self.NumClass = len(self.labels)

        self.ExpOut = numpy.array([None for i in range(self.SamplesNum)])

        for i in range(self.SamplesNum):
            self.ExpOut[i] = {}
            for label in self.labels:
                if label != self._Label[i]:
                    self.ExpOut[i][label] = -1.0/(self.NumClass - 1)
                else:
                    self.ExpOut[i][label] = +1.0

        # Make sure that the inputed data's demention is right.
        assert self.SamplesNum == self._Label.size

        self.Weaker = WeakerClassifier

        self.Discrete = Discrete

        self.W = {}
        for i in range(self.SamplesNum):
            val = toHashableVal(self._Mat[:, i])
            self.W[ val ] = 1.0/self.SamplesNum

        self.N = 0
        self.H = {}

        self.probability = {}

    def train(self, M = 100000):
	"""
	function @train() is the main process which run 
	AdaBoost algorithm.

	@M : Upper bound weaker classifier. How many weaker 
        classifier will be used to construct a strong 
	classifier.
	"""
        K = self.NumClass * 1.0
        factor = ((K-1)**2)/K

        for m in range(M-1):
            self.H[m] = self.Weaker(self._Mat, self._Label, 
                                    Discrete = self.Discrete, 
                                    Depth = 1, W = self.W)

            self.H[m].train()

            output = self.H[m].prediction(self._Mat)

            proba = self.H[m].proba

            log_proba = {}

            # calculate log(proba).
            for i in range(self.SamplesNum):
                val = toHashableVal(self._Mat[:, i])
                log_proba[val] = {}
                for label in self.labels:
                    if proba[val][label] < 1e-20:
                        log_proba[val][label] = numpy.log(1e-20)
                    else:
                        log_proba[val][label] = numpy.log(proba[val][label])

            self.probability[m] = {}

            # calculate H[m]k in papper
            for i in range(self.SamplesNum):
                val = toHashableVal(self._Mat[:, i])
                self.probability[m][val] = {}

                summer = 0.0
                for label in self.labels:
                    summer += log_proba[val][label]

                for label in self.labels:
                    self.probability[m][val][label] = \
                          (K - 1) * (log_proba[val][label] - summer)

            # update weight
            for i in range(self.SamplesNum):
                val = toHashableVal(self._Mat[:, i])

                # Y[i] * log(p[x[i]])
                summer = 0.0
                for label in self.labels:
                    summer += self.ExpOut[i][label] * log_proba[val][label]

                if self._Label[i] == output[i]:
                    # classified correctly
                    self.W[val] *= numpy.exp((-1.*(K-1)/K) * summer * (+1))
                else:
                    #misclassified
                    self.W[val] *= numpy.exp((-1.*(K-1)/K) * summer * (-1))
            if self.is_good_enough():
                print "It's good enough with ", self.N + 1, "weak classifier"
                break

            self.N += 1

    def is_good_enough(self):
        output = self.prediction(self._Mat)

        if numpy.all(output == self._Label):
            return True
        else:
            return False

    def prediction(self, Mat):

        VotingResult = {}

        Num = Mat.shape[1]

        for i in range(Num):
            val = toHashableVal(Mat[:, i])
            VotingResult[val] = {}
            for label in self.labels:
                VotingResult[val][label] = 0.0

        for m in range(self.N+1):
            proba = self.H[m].proba
            for i in range(Num):
                val = toHashableVal(Mat[:, i])

                for label in self.labels:
                    VotingResult[val][label] += proba[val][label]

        output = numpy.array([None for i in range(Num)])

        for i in range(Num):
            maxval = -numpy.inf
            val = toHashableVal(Mat[:, i])
            result = None
            for label in self.labels:
                if VotingResult[val][label] > maxval:
                    maxval = VotingResult[val][label]
                    result = label
            output[i] = result

        return output
