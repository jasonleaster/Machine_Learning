"""
Programmer  :   EOF
Date        :   2015.12.10
File        :   tester.py
E-mail      :   jasonleaster@163.com

Test case come from chapter 3 in <<Statistic methods>> By Hang-Li.
"""
import numpy
from knn import K_Nearest_Neighbor

# k == 2 two demention data
Original_Data = numpy.array([
[2,3],
[5,4],
[9,6],
[4,7],
[8,1],
[7,2]]).transpose()

a = K_Nearest_Neighbor(Original_Data)

print a.search([3, 4.5], a.tree)
