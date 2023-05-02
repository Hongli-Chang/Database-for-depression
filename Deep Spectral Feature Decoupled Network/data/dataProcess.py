'''
2018.12.03
@lsy

Database: SEED
Function:

'''

from __future__ import absolute_import
from __future__ import division 
from __future__ import print_function 
from sklearn.model_selection import train_test_split
from tensorflow.keras import utils as np_utils
import numpy as np 
import scipy.io as scio 
import os 

class DataGenerate:
    def __init__(self, data, label, subject, testSub):
        self.data = data
        self.label = label
        self.subject = subject
        self.testSub = testSub
        self.divideTrainTest()
        #self.dataPreprocess()
        self.shuffleData()


    def divideTrainTest(self):       
        '''
        Divide data into train data and test data.
        '''
        idx = [i for i in range(self.label.shape[1]) if self.subject[i] != self.testSub]
        self.train_data = self.data[:,:,idx]
        self.train_label = self.label[:,idx]

        idx = []
        idx = [i for i in range(self.label.shape[1]) if self.subject[i] == self.testSub]
        self.test_data = self.data[:,:,idx]
        self.test_label = self.label[:,idx]

    def shuffleData(self):
        idx = np.random.permutation(len(self.train_label[1,:]))
        self.train_data = self.train_data[:,:,idx]
        self.train_label = self.train_label[:,idx]








