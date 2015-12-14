"""
Programmer  :   EOF
E-mail      :   jasonleaster@163.com
File        :   svm.py
Date        :   2015.12.13

"""

import numpy

class SVM:
    def __init__(self, Mat, Tag, C = 10):
        self._Mat = numpy.array(Mat)
        self._Tag = numpy.array(Tag).flatten()

        self.SampleDem = self._Mat.shape[0]
        self.SampleNum = self._Mat.shape[1]

        self.C = C

        self.alpha = numpy.array([0.0 for i in range(self.SampleNum)])
        self.W     = numpy.array([0.0 for i in range(self.SampleDem)])
        self.b     = 0.0

        self.E     = numpy.array([0.0 for i in range(self.SampleNum)])

        self.Kernel = self.Linear_Kernel

        self.SupVec = []

    """
    Inner product of point @i and @j.
    K(i,j)
    """
    def Linear_Kernel(self, i, j):
        summer = 0.0
        for d in range(self.SampleDem):
            summer += self._Mat[d][i] * self._Mat[d][j]

        return summer

    def G(self, i):
        summer = 0.0
        for j in range(self.SampleNum):
            summer += self.alpha[j] * self._Tag[j] * self.Kernel(i,j)

        summer += self.b

        return summer

    """
    update the cost for prediction when x-i(Mat[:, i]) as input.
    """
    def updateE(self, i):
        self.E[i] = self.G(i) - self._Tag[i]

    def findFirstVar(self):
        firstPointIndex = None
        b_KKTcond_Points = []
        
        for i in range(self.SampleNum):
            self.updateE(i)

        for i in range(self.SampleNum):
            if 0 < self.alpha[i] and self.alpha[i] < self.C:
                if self.G(i) * self._Tag[i] != 1:
                    b_KKTcond_Points.append(i)

        # if there is not point on the boundary break the KKT-condition
        if len(b_KKTcond_Points) == 0:
            for i in range(self.SampleNum):
                if self.alpha[i] == 0 and self._Tag[i] * self.G(i) < 1:
                    b_KKTcond_Points.append(i)

                elif self.alpha[i] == self.C and self._Tag[i] * self.G(i) > 1:
                    b_KKTcond_Points.append(i)

        maxE = 0.0
        for i in b_KKTcond_Points:
            if abs(maxE) < abs(self.E[i]):
                firstPointIndex = i
                maxE = self.E[i]

        return firstPointIndex

    def findSecondVar(self, firstPointIndex):
        secondPointIndex = None

        val = 0
        if self.E[firstPointIndex] < 0:
            maxVal = self.E[firstPointIndex]
            for i in range(self.SampleNum):
                if self.E[i] > maxVal:
                    maxVal = self.E[i]
                    secondPointIndex = i
        else:
            minVal = self.E[firstPointIndex]
            for i in range(self.SampleNum):
                if self.E[i] < minVal:
                    minVal = self.E[i]
                    secondPointIndex = i

        return secondPointIndex


    """
        @optimal() function will update the alpha value of the 
    two selected points which could be indexed by @P1 and @P2.

        @P1 and @P2 are index of the first selected point 
    and the second selected point. You can get the point 
    by self._Mat[:, P1] and self._Mat[:, P2]
    """
    def optimal(self, P1, P2):
        if self._Tag[P1] == self._Tag[P2]:
            k = self.alpha[P2] - self.alpha[P1]
            L = max(0, k)
            H = min(self.C, self.C + k)
        else:
            k = self.alpha[P2] + self.alpha[P1]
            L = max(0, k - self.C)
            H = min(self.C, k)

        K11 = self.Kernel(P1, P1)
        K22 = self.Kernel(P2, P2)
        K12 = self.Kernel(P1, P2)

        miu = K11 + K22 - 2*K12

        old_alpha_P1 = self.alpha[P1]
        old_alpha_P2 = self.alpha[P2]

        # new alpha_2
        new_alpha_unc_P2 = old_alpha_P2 + \
        (self._Tag[P2] * (self.E[P1] - self.E[P2]) /miu)


        if self.alpha[P2] > H:
            new_alpha_P2 = H
        elif self.alpha[P2] < L:
            new_alpha_P2 = L
        else:
            new_alpha_P2 = new_alpha_unc_P2
            

        new_alpha_P1 = old_alpha_P1 + self._Tag[P1] * self._Tag[P2] * \
                (old_alpha_P2 - new_alpha_P2)

        self.alpha[P1] = new_alpha_P1
        self.alpha[P2] = new_alpha_P2


        """
        update b
        """
        new_E_P1 = 0
        for i in range(self.SampleNum):
            if i == P1 or i == P2:
                continue

            new_E_P1 += self.alpha[i] * self._Tag[i] * self.Kernel(i, P1)

        new_E_P1 += old_alpha_P1 * self._Tag[P1] * self.Kernel(P1, P1)
        new_E_P1 += old_alpha_P2 * self._Tag[P2] * self.Kernel(P2, P1)
        new_E_P1 += self.b - self._Tag[P1]

        
        new_E_P2 = 0
        for i in range(self.SampleNum):
            if i == P1 or i == P2:
                continue

            new_E_P2 += self.alpha[i] * self._Tag[i] * self.Kernel(i, P2)

        new_E_P2 += old_alpha_P1 * self._Tag[P1] * self.Kernel(P1, P2)
        new_E_P2 += old_alpha_P2 * self._Tag[P2] * self.Kernel(P2, P2)
        new_E_P2 += self.b - self._Tag[P2]

        b_P1_new = -new_E_P1 - self._Tag[P1] * self.Kernel(P1, P1) * (new_alpha_P1 - old_alpha_P1) \
                             - self._Tag[P2] * self.Kernel(P2, P1) * (new_alpha_P2 - old_alpha_P2) + self.b

        b_P2_new = -new_E_P2 - self._Tag[P1] * self.Kernel(P1, P2) * (new_alpha_P1 - old_alpha_P1) \
                             - self._Tag[P2] * self.Kernel(P2, P2) * (new_alpha_P2 - old_alpha_P2) + self.b


        if 0 < new_alpha_P1 and new_alpha_P1 < self.C and \
           0 < new_alpha_P2 and new_alpha_P2 < self.C:
            assert b_P1_new == b_P2_new

        if new_alpha_P1 == 0 or new_alpha_P1 == self.C or \
           new_alpha_P2 == 0 or new_alpha_P2 == self.C:
                self.b = (b_P1_new + b_P2_new)/2
        else:
            self.b = b_P1_new

        self.E[P1] = new_E_P1
        self.E[P2] = new_E_P2
        self.alpha[P1] = new_alpha_P1
        self.alpha[P2] = new_alpha_P2

        for i in range(self.SampleNum):
            summer = 0.0
            for j in range(self.SampleNum):
                if 0 < self.alpha[j] and self.alpha[j] < self.C:
                    # self.SupVec.append(i)
                    summer += self._Tag[j] * self.alpha[j] * self.Kernel(i, j) - self._Tag[i]
            self.E[i] = summer    

    def train(self):
        while self.run_or_not():
            P1 = self.findFirstVar()
            P2 = self.findSecondVar(P1)

            self.optimal(P1, P2)

        for i in range(self.SampleNum):
            self.W += self.alpha[i] * self._Mat[:, i] * self._Tag[i]

        for j in range(self.SampleNum):
            if 0 < self.alpha[j] and self.alpha[j] < self.C:
                summer = 0.0
                for i in range(self.SampleNum):
                    summer += self._Tag[i] * self.alpha[i] * self.Kernel(i, j)
                self.b = self._Tag[j] - summer

                print "Congradulation! Traning finished successfully."
                break

    def run_or_not(self):

        summer = 0.0
        for i in range(self.SampleNum):
            summer += self.alpha[i] * self._Tag[i]
        if summer != 0:
            return True

        for i in range(self.SampleNum):
            if self.alpha[i] < 0 or self.alpha[i] > self.C:
                return True

        for i in range(self.SampleNum):
            if self.alpha[i] == 0:
                if self._Tag[i] * self.G(i) < 1:
                    return True
            elif self.alpha[i] == self.C:
                if self._Tag[i] * self.G(i) > 1:
                    return True
            elif 0 < self.alpha[i] and self.alpha[i] < self.C:
                if self._Tag[i] * self.G(i) != 1:
                    return True
        
        return False
