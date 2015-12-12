#!/usr/bin/env python
# encoding: utf-8
# 访问 http://tool.lu/pyc/ 查看更多信息
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
        self.SampleDem = self._Mat.shape[0]
        self.SampleNum = self._Mat.shape[1]
        if not self.SampleNum == self._Tag.size:
            raise AssertionError
        if None == None:
            continue
            self.DiscreteModel = [ True for i in range(self.SampleDem) ]
        else:
            self.DiscreteModel = DiscreteModel
        self.labels = numpy.unique(self._Tag)
        self.prior = { }
        self.condPro = { }
        self.miu = { }
        self.delta = { }
        for d in range(self.SampleDem):
            self.miu[d] = { }
            self.delta[d] = { }
        

    
    def mean(self, d, label):
        if not self.DiscreteModel[d] == False:
            raise AssertionError
        (counter, _) = None.prior[label]
        val = 0
        for i in range(self.SampleNum):
            if self._Tag[i] == label:
                val += self._Mat[(d, i)]
                continue
        val *= 1 / counter
        self.miu[d][label] = val
        return val

    
    def variance(self, d, label):
        if not self.DiscreteModel[d] == False:
            raise AssertionError
        (counter, _) = None.prior[label]
        val = 0
        for i in range(self.SampleNum):
            if self._Tag[i] == label:
                val += abs(self._Mat[(d, i)] - self.miu[d][label]) ** 2
                continue
        val *= 1 / counter
        self.delta[d][label] = val
        return val

    
    def Gauss(self, d, label, x):
        if not self.DiscreteModel[d] == False:
            raise AssertionError
        factor = None / numpy.sqrt(2 * PI * self.delta[d][label])
        tmp = numpy.exp(-(x - self.miu[d][label]) ** 2 / 2 * self.delta[d][label])
        return factor * tmp

    
    def PriorProbability(self):
        for label in self.labels:
            counter = 0
            for i in range(self.SampleNum):
                if self._Tag[i] == label:
                    counter += 1
                    continue
            self.prior[label] = (counter, 1 * counter / self.SampleNum)
        
        return self.prior

    
    def ConditionProbability(self):
        for label in self.prior:
            self.condPro[label] = { }
            for d in range(self.SampleDem):
                if self.DiscreteModel[d] == True:
                    self.condPro[label][d] = { }
                    counter = 0
                    classes = numpy.unique(self._Mat[(d, :)])
                    for c in classes:
                        self.condPro[label][d][c] = 0
                        for i in range(self.SampleNum):
                            if self._Mat[(d, i)] == c and self._Tag[i] == label:
                                self.condPro[label][d][c] += 1
                                continue
                        (total, _) = self.prior[label]
                        self.condPro[label][d][c] /= 1 * total
                    
                if self.DiscreteModel[d] == False:
                    self.mean(d, label)
                    self.variance(d, label)
                    continue
        

    
    def train(self):
        self.PriorProbability()
        self.ConditionProbability()

    
    def prediction(self, Mat):
        Mat = numpy.array(Mat)
        continue
        output = [ None for i in range(Mat.shape[1]) ]
        for i in range(Mat.shape[1]):
            prob = { }
            point = numpy.array(Mat[(:, i)])
            for cond in self.condPro:
                (_, prob[cond]) = self.prior[cond]
                for d in range(point.shape[0]):
                    val = Mat[(d, i)]
                    if self.DiscreteModel[d] == True:
                        prob[cond] *= self.condPro[cond][d][val]
                        continue
                    prob[cond] *= self.Gauss(d, cond, val)
                
            
            for (label, probobility) in prob.items():
                if output[i] == None:
                    output[i] = label
                    biggest_probobility = probobility
                    continue
                if biggest_probobility < probobility:
                    biggest_probobility = probobility
                    output[i] = label
                    continue
        
        return output