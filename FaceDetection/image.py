"""
Programmer  :   EOF
File        :   init_training_set.py
Date        :   2015.12.29
E-mail      :   jasonleaster@163.com

Description :
    This script file will initialize the image set
and read all images in directory which is given by
user.

"""
import numpy
import cv2
import os
from matplotlib import pyplot

class Image:
    def __init__(self, fileName = None, label = None):
        self.imgName = fileName
        self.img     = cv2.imread(fileName, cv2.IMREAD_GRAYSCALE)
        self.label   = label

        self.Row     = self.img.shape[0]
        self.Col     = self.img.shape[1]

        self.stdImg  = self.__normalization()

        self.iimg    = self.__integrateImg()

        self.haarA   = self.__calFeatures(feature = "A")
        self.haarB   = self.__calFeatures(feature = "B")
        self.haarC   = self.__calFeatures(feature = "C")
        self.haarD   = self.__calFeatures(feature = "D")
        self.haarE   = self.__calFeatures(feature = "E")

    def __integrateImg(self):
        image = self.stdImg

        #@iImg is integrated image of normalized image @self.stdImg
        iImg = numpy.array([ [0. for i in range(self.Col + 1)] 
                                 for j in range(self.Row + 1)])

        for i in range(1, self.Row + 1):
            for j in range(1, self.Col + 1):
                iImg[i][j] =     \
                iImg[i  ][j-1] + \
                iImg[i-1][j  ] - \
                iImg[i-1][j-1] + \
                image[i-1][j-1]

        return iImg

    def __normalization(self):
        image = self.img

        #@nImag normalized image
        stdImg = numpy.array([ [0. for i in range(self.Col)] 
                                   for j in range(self.Row)])
        sigma = 0.
        for i in range(self.Row):
            for j in range(self.Col):
                sigma += image[i][j]

        meanVal = sigma / (self.Row * self.Col)

        for i in range(self.Row):
            for j in range(self.Col):
                stdImg[i][j] = (image[i][j] - meanVal) / numpy.var(image)

        return stdImg

    """
    Types of Haar-like rectangle features

     --- ---      --- ---
    |   |   |    |   +   |
    | + | - |    |-------|
    |   |   |    |   -   |
     --- ---      -------
       A            B

     -- -- --     -------
    |  |  |  |   |___-___|       
    |- | +| -|   |___+___|
    |  |  |  |   |   -   |
     -- -- --     -------
        C            D

     --- ---
    | - | + |
    |___|___|
    | + | - |
    |___|___|
        E
    """
    def rectangleSum(self, X, Y):
        x0, x1 = X[0], X[1]
        y0, y1 = Y[0], Y[1]

        return self.iimg[x1][y1] - self.iimg[x0][y1] -\
               self.iimg[x1][y0] + self.iimg[x0][y0]


    def __calFeatures(self, feature = "A"):
        retVal = []

        if feature == "A" or feature == "B":
            divide_x, divide_y = 2, 1
        elif feature == "C" or feature == "D":
            divide_x, divide_y = 3, 1
        elif feature == "E" or feature == "F":
            divide_x, divide_y = 2, 2

        for i in range(self.Row - 2):
            for j in range(self.Col - 1):
                for yscale in range(1, ((self.Col - j)/ divide_y) - 1):
                    y0 = j
                    y1 = j + yscale
                    for xscale in range(1, ((self.Row - i)/ divide_x) - 1):
                        x0 = i
                        x1 = i + xscale
                        x2 = i + xscale * 2

                        if feature == "A":

                            retVal.append(self.rectangleSum([x0, x1], [y0, y1]) -
                                          self.rectangleSum([x1, x2], [y0, y1]) )
                        elif feature == "B":

                            retVal.append(self.rectangleSum([y0, y1], [x0, x1]) -
                                          self.rectangleSum([y0, y1], [x1, x2]) )

                        elif feature == "C":
                            x3 = i + xscale * 3

                            retVal.append(self.rectangleSum([x1, x2], [y0, y1]) -
                                          self.rectangleSum([x0, x1], [y0, y1]) -
                                          self.rectangleSum([x2, x3], [y0, y1]) )

                        elif feature == "D":
                            x3 = i + xscale * 3

                            retVal.append(self.rectangleSum([y0, y1], [x1, x2]) -
                                          self.rectangleSum([y0, y1], [x0, x1]) -
                                          self.rectangleSum([y0, y1], [x2, x3]) )

                        elif feature == "E":
                            y2 = j + yscale *2

                            retVal.append(self.rectangleSum([x0, x1], [y0, y1]) +
                                          self.rectangleSum([x1, x2], [y1, y2]) -
                                          self.rectangleSum([x1, x2], [y0, y1]) -
                                          self.rectangleSum([x0, x1], [y1, y2]) )

        retVal.sort()

        return retVal

    def show(self):
        cv2.namedWindow(self.imgName)
        cv2.imshow(self.imgName, self.img)
        cv2.waitKey(0)
        cv2.destroyWindow(self.imgName)

    def showHaarFeature(self, haarFeat = "A"):
        
        if haarFeat == "A":
            feature = self.haarA
        elif haarFeat == "B":
            feature = self.haarB 

        for i in range(0, len(self.haarA), 100):
            pyplot.plot(i, self.haarA[i], "ob")
            pyplot.plot(i, self.haarB[i], "or")

        pyplot.show()

class ImageSet:
    def __init__(self, imgDir = None, label = None, sampleNum = None):

        assert isinstance(imgDir, str)

        self.fileList = os.listdir(imgDir)
        self.fileList.sort()

        if sampleNum == None:
            self.sampleNum = len(self.fileList)
        else:
            self.sampleNum = sampleNum

        self.setLabel  = label

        self.images = [None for i in range(self.sampleNum)]
        processed = -10.
        for i in range(self.sampleNum):
            self.images[i] = Image(imgDir + self.fileList[i], label)

            if i % (self.sampleNum / 10) == 0:
                processed += 10.
                print "Loading ", processed, "%"

        print "Loading  100 %\n"
        # Haar Featrue Groups
        self.haarFeatGroups = []
        #self.featureNum = self.__countFeatures()

    """
    def __countFeatures(self, ImgSize):
        Row, Col = ImgSize

        s1, s2 = 0, 0
        for j in range(2, Col + 1): s1 += j/2
        for i in range(1, Row + 1): s2 += i/1

        # For feature A and B
        sumAB = s1 * s2 * 2

        s1, s2 = 0, 0
        for j in range(3, Col + 1): s1 += j/3
        for i in range(1, Row + 1): s2 += i/1

        # For feature C and D
        sumCD = s1 * s2 * 2

        s1, s2 = 0, 0
        for j in range(2, Col + 1): s1 += j/2
        for i in range(2, Row + 1): s2 += i/2

        sumE = s1 * s2

        return sumAB + sumCD + sumE
    """
