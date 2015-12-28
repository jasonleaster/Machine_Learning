"""
Programmer  :   EOF
E-mail      :   jasonleaster@163.com
File        :   decisionTree.py
Date        :   2015.12.08

Usage:
    1. Create a decision tree with constructor 
    DT = DecisionTree(Mat, Tag, Discrete)

    2.Training this decision tree with member function DT.train()

    3.Prediction. DT.prediction(Test_Case)

    If you are user who have no idea about what a monster this is, 
    just ignore the implementation and call this three API is OK :)
"""

import numpy
from tree import Tree

"""
helper function of @DecisionTree.Gini()
This function take a list @labels as an argument.

According to equation:
    Gini(p) == sum all {p_m*(1 - p_m)} where (m belongs to [1 to M]) 

    p_m == (number of point of type m) / (number of all points)

    M: how many different type of inputed data of @pGini()
        In this function, M is len(label_set)
"""
def pGini(labels):

    label_set = set(labels)
    summer = 0.0
    Total = labels.shape[0] * 1.0
    for i in label_set:
        summer += (numpy.count_nonzero(labels == i)/Total)**2

    return 1 - summer

def toHashableVal(Vec):
    val = ""
    for i in range(len(Vec)):
        val += str(Vec[i]) 

    return val

class DecisionTree:
    def __init__(self, Mat, Tag, Discrete = None, Depth = None, W=None):
        """
        @Mat: Inputed data points which represent as a matrix.
              Every column vector in @Mat is a feature of training set
        @Tag: Labels with training points.

        @Discrete: It's a bool vector of size @SamplesNumber
            Discrete[i] == 0 means the i-th feature is discrete 
            feature, otherwise it's continuous
            In default, all feature is discrete

        """
        self._Mat = numpy.array(Mat)
        self._Tag = numpy.array(Tag).flatten()

        """
        @SamplesNum: how many sample points in inputed training set
        @SamplesDem: the demention of inputed point which 
                     **also** means how many features in inputed data.
        """
        self.SamplesDem = self._Mat.shape[0]
        self.SamplesNum = self._Mat.shape[1]

        self.feature_dict = {}
        self.labels = numpy.unique(self._Tag)

        self.DT = Tree()

        if Discrete == None:
            self.Discrete = [True for i in range(self.SamplesDem)]
        else:
            self.Discrete = Discrete

        for i in range(self.SamplesDem):
            if self.Discrete[i] == True:
                self.feature_dict[i] = numpy.unique(self._Mat[i, :])

        if self.SamplesNum != self._Tag.size:
            print "Error: Make sure that the number of tag ",\
                  "is same as points in inputed data"

        self.limitedDepth = Depth
        self.currentDepth = 0

        assert Depth >= 1

        if W == None:
            self.W = {}
            for i in range(self.SamplesNum):
                val = toHashableVal(self._Mat[:, i])
                self.W[ val ] = 1.0/self.SamplesNum
        else:
            assert isinstance(W, dict)
            self.W = W

    def Gini(self, Vec, Tag, val, feature):
        # c1 + c2 == D
        if self.Discrete[feature] == True:
            c1 = 1.0 * numpy.count_nonzero(Vec == val)
            c2 = 1.0 * numpy.count_nonzero(Vec != val)
            D =  1.0 * Tag.size

            return c1 * pGini(Tag[ Vec == val]) / D + \
                   c2 * pGini(Tag[ Vec != val]) / D
        else:
            c1 = 1.0 * numpy.count_nonzero(Vec >= val)
            c2 = 1.0 * numpy.count_nonzero(Vec <  val)
            D =  1.0 * Tag.size

            return c1 * pGini(Tag[ Vec >= val]) / D + \
                   c2 * pGini(Tag[ Vec <  val]) / D

    """
    Function @makeTree():
        this function will create a decision tree recursively.

    @opt_feature : optimal feature of current decision process
    @opt_val     : optimal val in @opt_feature which is used 
                   to classify current data set
    @miniumn     : minumn value of Gini-cost
    """
    def makeTree(self, Mat, Tag):
        miniumn     = + numpy.inf
        opt_feature = 0
        opt_val     = 0

        Tag = numpy.array(Tag)

        #return the type of data, if there is no different type of data
        if numpy.unique(Tag).size == 1:
            t = Tree()
            t.isLeaf  = True
            t.nodeVal = Tag[0]
            
            for label in self.labels:
                if label != Tag[0]:
                    t.counter[label] = 0.0
                else:
                    t.counter[label] = 1.0
            return t

        minium = + numpy.inf
        for f in range(self.SamplesDem):
            for i in range(len(Tag)):
                v = Mat[f, i]
                p = self.Gini(Mat[f], Tag, v, f)
                p /= self.W[toHashableVal(Mat[:, i])]
                if p < miniumn:
                    miniumn     = p
                    opt_feature = f
                    opt_val     = v

        t = Tree()
        t.nodeVal    = opt_val
        t.selFeature = opt_feature

        if self.currentDepth == self.limitedDepth:
            t.isLeaf = True

            for label in self.labels:
                t.counter[label] = 0

            summer = 0.0
            for i in range(len(Tag)):
                label = Tag[i]
                t.counter[label] += self.W[toHashableVal(Mat[:, i])]
                summer += self.W[toHashableVal(Mat[:, i])]

            for label in numpy.unique(Tag):
                t.counter[label] /= summer
            
            return t

        if miniumn == 1:
            return Tag[0]

        self.currentDepth += 1

        if self.Discrete[opt_feature] == True:
            t.left  = self.makeTree(
                    Mat[:, Mat[opt_feature, :] == opt_val], 
                    Tag[   Mat[opt_feature, :] == opt_val])

            t.right = self.makeTree(
                    Mat[:, Mat[opt_feature, :] != opt_val], 
                    Tag[   Mat[opt_feature, :] != opt_val])
        else:
            t.left  = self.makeTree(
                    Mat[:, Mat[opt_feature, :] <  opt_val], 
                    Tag[   Mat[opt_feature, :] <  opt_val])

            t.right = self.makeTree(
                    Mat[:, Mat[opt_feature, :] >= opt_val], 
                    Tag[   Mat[opt_feature, :] >= opt_val])

        return t

    def train(self):
        self.DT = self.makeTree(self._Mat, self._Tag)

        self.proba = self.getProba(self._Mat)

    def getProba(self, Mat):
        Dem = Mat.shape[0]
        Num = Mat.shape[1]

        proba = {}

        for i in range(Num):
            node = self.DT
            while node.isLeaf != True:
                if self.Discrete[node.selFeature] == False:
                    if Mat[node.selFeature, i] >= node.nodeVal:
                        node = node.right
                    else:
                        node = node.left
                else:
                    if Mat[node.selFeature, i] != node.nodeVal:
                        node = node.right
                    else:
                        node = node.left


            val = toHashableVal(Mat[:, i])
            proba[val] = {}
            for label in self.labels:
                proba[val][label] = node.counter[label]

        return proba
            

    def prediction(self, Mat):

        probability = self.proba

        Num = Mat.shape[1]
        Dem = Mat.shape[0]

        output = numpy.array([None for i in range(Num)])

        for i in range(Num):
            maxval = -numpy.inf
            val = toHashableVal(Mat[:, i])
            result = None
            for label in self.labels:
                if probability[val][label] > maxval:
                    maxval = probability[val][label]
                    result = label
            output[i] = result

        return output
