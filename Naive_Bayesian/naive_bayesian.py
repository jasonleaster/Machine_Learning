'''
Programmer  :   EOF
E-mail      :   jasonleaster@163.com
Date        :   2015.12.12
File        :   naive_beyesian.py

'''
import numpy
PI = 3.14159

class NaiveBayesian:
    """
                   $ Key words explantion $

    Feature(demention):
             It also corresponding to dimentions of sample.
             eg. With point which is a vector X. X[d] means the d-th
                demention value in X.

    @self._Mat : The represent of inputed data. Vector Mat[:, i] 
                 is a sample point. All vector construct this matrix

    @self._Tag : The label of inputed sample points.

    @self.DiscreteModel: DiscreteModel[i] means that the i-th 
                feature(or demention) of points is a discrete
                variable if it's value is True.

                !! If the feature is continuous variable, we have to
                use Gauss Probobility Model to estimate the result.
    """
    
    def __init__(self, Mat, Tag, DiscreteModel = None):
        self._Mat = numpy.array(Mat)
        self._Tag = numpy.array(Tag).flatten()

        self.SampleDem = self._Mat.shape[0] # Demention of samples
        self.SampleNum = self._Mat.shape[1] # Number of samples

        assert self.SampleNum == self._Tag.size

        if DiscreteModel == None:
            self.DiscreteModel = [True for i in range(self.SampleDem)]
        else:
            self.DiscreteModel = DiscreteModel

        self.labels = numpy.unique(self._Tag)
        self.prior   = {} #prior probobility
        self.condPro = {} #condition probobility

        """
        This part data member is only used in continuous fearture
        """
        self.miu    = {} # mean value of continuous feature
        self.delta  = {} # variance of continuous feature
        for d in range(self.SampleDem):
            self.miu[d]   = {}
            self.delta[d] = {}
        

    
    def mean(self, d, label):
        assert self.DiscreteModel[d] == False

        (counter, _) = self.prior[label]
        val = 0
        for i in range(self.SampleNum):
            if self._Tag[i] == label:
                val += self._Mat[d, i]

        val *= (1.0 / counter)
        self.miu[d][label] = val
        return val

    
    def variance(self, d, label):
        assert self.DiscreteModel[d] == False
        (counter, _) = self.prior[label]

        val = 0
        for i in range(self.SampleNum):
            if self._Tag[i] == label:
                val += abs(self._Mat[d, i] - self.miu[d][label]) ** 2
        val *= 1.0/ counter
        self.delta[d][label] = val

        return val

    
    def Gauss(self, d, label, x):
        assert self.DiscreteModel[d] == False
        
        factor = 1.0/ numpy.sqrt(2 * PI * self.delta[d][label])
        tmp = numpy.exp(-(x - self.miu[d][label]) ** 2 / \
                        ( 2 * self.delta[d][label]))
        return factor * tmp

    
    def PriorProbability(self):
        for label in self.labels:
            counter = 0
            for i in range(self.SampleNum):
                if self._Tag[i] == label:
                    counter += 1
            self.prior[label] = (counter, \
                                (1.0 * counter) / self.SampleNum)
        
        return self.prior

    
    def ConditionProbability(self):
        for label in self.prior:
            self.condPro[label] = { }

            for d in range(self.SampleDem):

                if self.DiscreteModel[d] == True:

                    self.condPro[label][d] = {}

                    counter = 0
                    classes = numpy.unique(self._Mat[d, :])
                    for c in classes:

                        self.condPro[label][d][c] = 0
                        for i in range(self.SampleNum):
                            if self._Mat[(d, i)] == c and self._Tag[i] == label:

                                self.condPro[label][d][c] += 1

                        (total, _) = self.prior[label]
                        self.condPro[label][d][c] /= (1.0 * total)
                    
                if self.DiscreteModel[d] == False:
                    self.mean(d, label)
                    self.variance(d, label)
        
    """
    All user just call this two API is ok, if you don't want to know
    the detail about how to implement this algorithm
    """
    def train(self):
        self.PriorProbability()
        self.ConditionProbability()

    """
    Make sure that your input @Mat construct like this:
        Mat = [Vec1, Vec2 ... VecN]. Each Vec* is sample point vector.
        you can get the i-th point by Mat[:, i]
    """
    def prediction(self, Mat):
        Mat = numpy.array(Mat)

        output = [ None for i in range(Mat.shape[1]) ]

        for i in range(Mat.shape[1]):
            prob = {}
            point = numpy.array(Mat[:, i])
            for cond in self.condPro: # @cond means labels in samples

                (_, prob[cond]) = self.prior[cond]
                for d in range(point.shape[0]):

                    val = Mat[(d, i)]
                    if self.DiscreteModel[d] == True:
                        prob[cond] *= self.condPro[cond][d][val]
                    else:
                        prob[cond] *= self.Gauss(d, cond, val)
                
            
            for (label, probobility) in prob.items():
                if output[i] == None:
                    output[i] = label
                    biggest_probobility = probobility

                if biggest_probobility < probobility:
                    biggest_probobility = probobility
                    output[i] = label
        
        return output
