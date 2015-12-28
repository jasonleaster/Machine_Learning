"""
Programmer  :   EOF
File        :   read_training_image.py
Date        :   2015.12.16
E-mail      :   jasonleaster@163.com
"""

import numpy
import cv2

def str2int(string, length):
    summer = 0
    for i in range(length):
        summer <<= 8
        summer += ord(string[i])

    return summer

class ImageSet:
    def __init__(self, imgFileName, labelFileName):
        
        """
        Initialization of image file which is used for training
        """
        training_set_img = open(imgFileName)

        Size_Magic_Number = 4 #4 byte
        Size_Num_of_img   = 4
        Size_Num_of_row   = 4
        Size_Num_of_col   = 4

        images_MagicNum = training_set_img.read(Size_Magic_Number)
        Num_of_img      = training_set_img.read(Size_Num_of_img)
        Num_of_row      = training_set_img.read(Size_Num_of_row)
        Num_of_col      = training_set_img.read(Size_Num_of_col)
        

        images_MagicNum = str2int(images_MagicNum, Size_Magic_Number)
        Num_of_img      = str2int(Num_of_img,   Size_Num_of_img)
        Num_of_row      = str2int(Num_of_row,   Size_Num_of_row)
        Num_of_col      = str2int(Num_of_col,   Size_Num_of_col)

        assert images_MagicNum == 2051

        #self.SampleNum = Num_of_img
        self.SampleNum = 1000
        self.Row       = Num_of_row
        self.Col       = Num_of_col
        self.Pixels    = self.Row * self.Col

        # we use a list @self.images to store the @Image object
        self.images = [None for i in range(self.SampleNum)]

        imgs_string = training_set_img.read(self.Row * self.Col * self.SampleNum)
        for n in range(self.SampleNum):

            imgMat = numpy.array([[0 for i in range(self.Col)] 
                                     for j in range(self.Row)])

            for i in range(self.Row):
                for j in range(self.Col):
                    imgMat[i][j] = ord(imgs_string[(self.Row * i + j) + n * self.Pixels])

            self.images[n] = Image(imgMat, self.Row, self.Col)


        """
        Initialization of label file which is used for training
        """
        training_set_label = open(labelFileName)

        Size_Magic_Num_label = 4
        Size_Num_of_label    = 4
        Size_label           = 4

        labels_MagicNum = training_set_label.read(Size_Magic_Number)
        Num_of_label    = training_set_label.read(Size_Num_of_label)

        labels_MagicNum = str2int(labels_MagicNum, Size_Magic_Number)
        #Num_of_label    = str2int(Num_of_label, Size_Num_of_label)
        Num_of_label    = 1000


        assert labels_MagicNum == 2049
        assert Num_of_label == self.SampleNum

        self.labels = [None for i in range(self.SampleNum)]

        labels_string = training_set_label.read(Size_label * self.SampleNum)
        for i in range(self.SampleNum):
            self.labels[i] = ord(labels_string[i:i+1])

        print "Initialization of training set image",\
              " finished successfully 2333"


    def flattenImg(self, i):
        return self.images[i].Img.flatten()

class Image:
    def __init__(self, Img, Row, Col):
        self.Img = Img
        self.Row = Row
        self.Col = Col
        self.name = "A Hand Written Image"

        self.inteImg = self.haarfeature()

    def show(self):
        cv2.namedWindow(self.name)
        cv2.imshow(self.name, self.Img/255.)
        cv2.waitKey(0)
        cv2.destroyWindow(self.name)

    def haarfeature(self):
        image = self.Img
        integratedImg = [ [0 for i in range(self.Col)]
                             for j in range(self.Row)]

        """
        for i in range(self.Row):
            for j in range(self.Col):
                integratedImg[i][j] = integratedImg[i-1][j-1] + \
                                      image[i-1][j  ] + \
                                      image[i  ][j-1] + \
                                      image[i  ][j  ]
        """

        for i in range(self.Row):
            for j in range(self.Col):
                integratedImg[i][j] = integratedImg[i  ][j-1] + \
                                      integratedImg[i-1][j  ] - \
                                      integratedImg[i-1][j-1] + \
                                              image[i-1][j-1]



