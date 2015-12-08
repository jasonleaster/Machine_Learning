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

    If you a user and have no idea about what a monster this is, 
    just ignore the implementation and call this three API is OK :)
"""

import numpy

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

class DecisionTree:
    def __init__(self, Mat, Tag, Discrete = None):
        """
        @Mat: Inputed data points which represent as a matrix.
              Every column vector in @Mat is a feature of training set
        @Tag: Labels with training points.

        @Discrete: It's a bool vector of size @SamplesNumber
            Discrete[i] == 0 means the i-th feature is discrete 
            feature, otherwise it's continuous
            In default, all feature is discrete

        """
        self._Mat = numpy.array(Mat).transpose()
        self._Tag = numpy.array(Tag).flatten(1)

        """
        @SamplesNumber: how many sample points in inputed training set
        @SamplesDemention: the demention of inputed point which 
                            **also** means how many features in 
                            inputed data.
        """
        self.SamplesNumber, self.SamplesDemention = self._Mat.shape

        self.feature_dict = {}
        self.labels = numpy.unique(Tag)

        """
        @DT : we recursively define the node of decision tree with this             pattern -->
            [(current_best_feature, current_best_val_in_feature), 
             right_sub_tree, 
             left_sub_tree]

             sub_tree is a leaf, if it's label of inputed data, which
             means that leaf breaks this the definition of node.
        """
        self.DT = []
        if Discrete == None:
            self.Discrete = [True for i in range(self.SamplesDemention)]
        else:
            self.Discrete = Discrete

        for i in range(self.SamplesDemention):
            self.feature_dict[i] = numpy.unique(self._Mat[:, i])

        if self.SamplesNumber != self._Tag.size:
            print "Error: Make sure that the number of tag ",\
                  "is same as points in inputed data"


    def Gini(self, Mat, Tag, feature, val):
        # c1 + c2 == D
        if self.Discrete[feature] == True:
            c1 = 1.0 * (Mat[ Mat[:, feature] == val ]).shape[0]
            c2 = 1.0 * (Mat[ Mat[:, feature] != val ]).shape[0]
            D =  1.0 * Tag.shape[0]

            return c1 * pGini(Tag[ Mat[:, feature] == val]) / D + \
                   c2 * pGini(Tag[ Mat[:, feature] != val]) / D
        else:

            c1 = 1.0 * (Mat[ Mat[:, feature] >= val ]).shape[0]
            c2 = 1.0 * (Mat[ Mat[:, feature] <  val ]).shape[0]
            D  = 1.0 * Tag.shape[0]

            return c1 * pGini(Tag[ Mat[:, feature] >= val]) / D + \
                   c2 * pGini(Tag[ Mat[:, feature] <  val]) / D

    """
    Function @makeTree():
        this function will create a decision tree recursively.

    @opt_feature : optimal feature of current decision process
    @opt_val     : optimal val in @opt_feature which is used 
                   to classify current data set
    @miniumn     : minumn value of Gini-cost
    """
    def makeTree(self, Mat, Tag):
        miniumn     = 10000.0
        opt_feature = 0
        opt_val     = 0

        #return the type of data, if there is no different type of data
        if numpy.unique(Tag).size == 1:
            return Tag[0]

        for f in range(self.SamplesDemention):
            for v in self.feature_dict[f]:
                p = self.Gini(Mat, Tag, f, v)
                if p < miniumn:
                    miniumn     = p
                    opt_feature = f
                    opt_val     = v

        if miniumn == 1:
            return Tag[0]

        left = []
        right = []

        if self.Discrete[opt_feature] == True:
            left  = self.makeTree(Mat[ Mat[:, opt_feature] == opt_val], 
                                  Tag[ Mat[:, opt_feature] == opt_val])

            right = self.makeTree(Mat[ Mat[:, opt_feature] != opt_val], 
                                  Tag[ Mat[:, opt_feature] != opt_val])
        else:

            left  = self.makeTree(Mat[ Mat[:, opt_feature] >= opt_val], 
                                  Tag[ Mat[:, opt_feature] >= opt_val])

            right = self.makeTree(Mat[ Mat[:, opt_feature] <  opt_val], 
                                  Tag[ Mat[:, opt_feature] <  opt_val])

        return [(opt_feature, opt_val), left, right]

    def train(self):
        self.DT = self.makeTree(self._Mat, self._Tag)

    def prediction(self, Mat):
        if self.emptyDT():
            print "Error: empyt decision tree. Can not determine",\
                   "which class the inputed data is."
            return

        Mat = numpy.array(Mat).transpose()
        result = numpy.zeros((Mat.shape[0], 1))
        for i in range(Mat.shape[0]):
            tree = list(self.DT)

            while self.isLeaf(tree) == False:
                feature, val = tree[0]
                if self.Discrete[feature] == True:
                    if Mat[i][feature] == val:
                        tree = self.getLeft(tree)
                    else:
                        tree = self.getRight(tree)
                else:
                    if Mat[i][feature] >= val:
                        tree = self.getLeft(tree)
                    else:
                        tree = self.getRight(tree)

            result[i] = tree

        return result


    """
    helper function
    """
    def emptyDT(self):
        if len(self.DT) == 0:
            return True
        else:
            return False

    def isLeaf(self, tree):
        if isinstance(tree, list):
            return False
        else:
            return True

    def getOptMess(self, tree):
        assert isinstance(tree, list)
        return tree[0]

    def getLeft(self, tree):
        assert isinstance(tree, list)
        return tree[1]

    def getRight(self, tree):
        assert isinstance(tree, list)
        return tree[2]

