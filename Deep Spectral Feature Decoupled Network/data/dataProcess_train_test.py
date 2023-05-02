
from __future__ import absolute_import
from __future__ import division 
from __future__ import print_function 

import h5py
import numpy as np 
import scipy.io as scio 
import os 

class DataGenerate:
    def __init__(self, dataDir):
        self.dataDir = dataDir
        #self.pointer = pointer 
        self.dataRead()
        #self.divideTrainTest()
        #self.dataPreprocess()
        #self.batch = int(self.train_label.shape[0]/self.batch_size)
        self.shuffle_data()


    def dataRead(self):
        # data_info = scio.loadmat(self.dataDir)
        # self.data = np.asarray(data_info['data'], dtype=np.float32)            # [nsamples, nchannels, nfeature]
        # self.label = np.asarray(data_info['label'], dtype=np.int64)            # label = 0 ~ 2
        # self.label = self.label.reshape(self.label.shape[0])
        #self.video = np.asarray(data_info['video'], dtype=np.int64) - 1        # video = 0 ~ 14
        f = h5py.File(self.dataDir)
        self.train_data, self.train_label, self.test_data, self.test_label = f["train_X"], f["train_y_a"], f["test_X"], \
                                                           f["test_y_a"]

        self.train_data = self.train_data[()].reshape(self.train_data.shape[0], self.train_data.shape[1], 5)
        self.test_data = self.test_data[()].reshape(self.test_data.shape[0], self.train_data.shape[1], 5)
        #self.train_label =

        self.train_label = self.train_label[()].reshape(self.train_label.shape[0], self.train_label.shape[1])
        self.test_label = self.test_label[()].reshape(self.test_label.shape[0], self.test_label.shape[1])

        self.train_label = np.reshape(self.train_label[:,1], [self.train_label.shape[0]])
        self.test_label = np.reshape(self.test_label[:,1], [self.test_label.shape[0]])


    def shuffle_data(self):
        idx = np.random.permutation(len(self.train_label))
        train_data = self.train_data[idx, :, :]
        train_label = self.train_label[idx]
        test_data = self.test_data
        test_label = self.test_label

        return train_data, train_label, test_data, test_label







