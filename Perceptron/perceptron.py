"""
Programmer  :   EOF
Date        :   2015.12.10
File        :   perceptron.py
E-mail      :   jasonleaster@163.com

"""

import numpy

class Perceptron:
    """
    Perceptron is a classical linear classifier. @X is a vector of sample point.

    Model : f(x) = sign(W * X + b)
    Target: Using samples which have a tag or label to 
            get @W and @b by training.

    self._Mat: [[X1],[X2] ... [Xn]], where n is @self.SampleNum that the number
                of training samples.

    self.w : the weight vector
    self.b : bias (a constant number)
    """
    def __init__(self, Mat, Tag):
        self._Mat = numpy.array(Mat).transpose()
        self._Tag = numpy.array(Tag).transpose().flatten()

        self.SampleDem = self._Mat.shape[0]
        self.SampleNum = self._Mat.shape[1]

        self.w = numpy.array([0.0 for i in range(self.SampleDem)])
        self.b = 0.0

        self.miu = 1.0

    def dot_production(self, A, B):
        assert len(A) == len(B)
        summer = 0
        for i in range(len(A)):
            summer += A[i] * B[i]

        return summer

    def classify(self, i, w, b):
        if (self.dot_production(self._Mat[:, i], w) + b) *self._Tag[i] <= 0:
            return False
        else:
            return True


    def train(self):

        misclassfied = [True for i in range(self.SampleNum)]

        i       = 0
        counter = 0
        while True in misclassfied:

            if i == self.SampleNum:
                i = 0

            point = self._Mat[:, i]

            if self.classify(i, self.w, self.b) == False:
                self.w += self.miu * self._Tag[i] * point
                self.b += self.miu * self._Tag[i]

                counter += 1
                print "Iteration time: ", counter, "misclassified x", (i+1), "w: ", self.w, "b: ", self.b

                for j in range(self.SampleNum):
                    p = self._Mat[:, j]
                    if self.classify(j, self.w, self.b) == False:
                        misclassfied[j] = True
                    else:
                        misclassfied[j] = False
            else:
                i += 1

    
    def prediction(self, Mat):
        Mat = numpy.array(Mat)
        Dem = Mat.shape[0]
        Num = Mat.shape[1]

        output = []
        for i in range(Num):
            if self.dot_production(self.w, Mat[:,i]) + self.b > 0:
                output.append(+1)
            else:
                output.append(-1)

        return output
