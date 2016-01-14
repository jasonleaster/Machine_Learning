"""
Programmer  :   EOF
File        :   config.py
Date        :   2016.01.06
E-mail      :   jasonleaster@163.com

Description :
    This is a configure file for this project.

"""

TRAINING_FACE    = "./newtraining/face/"
TRAINING_NONFACE = "./newtraining/non-face/"
TEST_FACE        = "./newtraining/test/face/"
TEST_NONFACE     = "./newtraining/test/non-face/"

FEATURE_FILE_TRAINING = "./features/features_train.tmp"
FEATURE_FILE_TESTING  = "./features/features_test.tmp"

ADABOOST_FILE    = "./model.tmp"

FEATURE_TYPE_NUM = 2
FEATURE_NUM = 12546

POSITIVE_SAMPLE  = 140
NEGATIVE_SAMPLE  = 140

SAMPLE_NUM = POSITIVE_SAMPLE + NEGATIVE_SAMPLE

TESTING_POSITIVE_SAMPLE = 50
TESTING_NEGATIVE_SAMPLE = 50

TESTING_SAMPLE_NUM = TESTING_NEGATIVE_SAMPLE + TESTING_POSITIVE_SAMPLE
