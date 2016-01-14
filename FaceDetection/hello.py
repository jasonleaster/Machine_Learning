from matplotlib import pyplot as plt
from matplotlib import image
import pylab
import cv2

fileName = "./newtraining/face/face00004.pgm"

img = image.imread(fileName)
img1 = cv2.imread(fileName)

plt.matshow(img)
pylab.show()
