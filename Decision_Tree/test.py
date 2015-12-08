"""
Programmer  :   EOF
Date        :   2015.12.08
File        :   test.py
E-mail      :   jasonleaster@163.com

This is a test scrip of decision tree module
"""

import numpy
from decisionTree import *

#Original_Data = numpy.array([
#[0,0,0,0,8],
#[0,0,0,1,3.5],
#[0,1,0,1,3.5],
#[0,1,1,0,3.5],
#[0,0,0,0,3.5],
#[1,0,0,0,3.5],
#[1,0,0,1,3.5],
#[1,1,1,1,2],
#[1,0,1,2,3.5],
#[1,0,1,2,3.5],
#[2,0,1,2,3.5],
#[2,0,1,1,3.5],
#[2,1,0,1,3.5],
#[2,1,0,2,3.5],
#[2,0,0,0,10],
#]).transpose()

# This test data come from <<The methods of statistic>> by Hang Li
Original_Data = numpy.array([
['teenager',        'no',   'no',   0.0],
['teenager',        'no',   'no',   1.0],
['teenager',        'yes',  'no',   1.0],
['teenager',        'yes',  'yes',  0.0],
['teenager',        'no',   'no',   0.0],
['senior citizen',  'no',   'no',   0.0],
['senior citizen',  'no',   'no',   1.0],
['senior citizen',  'yes',  'yes',  1.0],
['senior citizen',  'no',   'yes',  2.0],
['senior citizen',  'no',   'yes',  2.0],
['old pepple',      'no',   'yes',  2.0],
['old pepple',      'no',   'yes',  1.0],
['old pepple',      'yes',  'no',   1.0],
['old pepple',      'yes',  'no',   2.0],
['old pepple',      'no',   'no',   0.0],
]).transpose()

Tag = numpy.array([
[-1],
[-1],
[+1],
[+1],
[-1],
[-1],
[-1],
[+1],
[+1],
[+1],
[+1],
[+1],
[+1],
[+1],
[-1],
]).transpose()

discrete = [ True for i in range(Original_Data.shape[0])]

# which feature's value is continue, we set 
# that discrete[index of that feature] = False
#discrete[4] = False

a = DecisionTree(Original_Data, Tag, discrete)

a.train()

print a.prediction(numpy.array([
    ['teenager',        'no',   'no',   0,  8],
    ['old pepple',      'yes',  'no',   2,3.5],
    ]).transpose())
