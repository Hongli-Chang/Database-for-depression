# -*- coding: utf-8 -*-
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras import utils as np_utils
import numpy as np
import h5py
import os.path
from sklearn.metrics import roc_auc_score
import pdb
import time
from keras.models import Sequential, Model
from keras.layers import Convolution2D, MaxPooling2D, Dense, Dropout, Flatten, InputLayer, Input, merge, concatenate, \
    add, Lambda, multiply
from keras import backend as K
from keras.utils.np_utils import to_categorical
from keras.layers.advanced_activations import PReLU, ELU, LeakyReLU
from scipy.io import savemat
from keras.models import Model
from keras.layers import Dense, Activation, Input, Reshape
from keras.layers import Conv1D, Flatten, Dropout
from keras.optimizers import SGD, Adam
from keras.layers.normalization import BatchNormalization
from scipy.io import loadmat
from keras.models import load_model
from model import reader_tensor
from model import reader_vector
from model import reader_tensor2
import time
import numpy as np
#import matplotlib.pyplot as plt
from itertools import cycle
from sklearn.metrics import confusion_matrix
from sklearn import svm, datasets
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from scipy import interp
from scipy.optimize import minimize_scalar

import pandas as pd
from data.dataProcess import DataGenerate

from data.gendata_seed_grouped_classes_Loso_lanzhou3 import gen_raw_data

from nets.nets_LSTM import input_model_lstm_deep as input_model

from nets.GD_basic import get_generative, get_discriminative

from util.util import label2uniqueID, split_test_as_valid, argmin_mean_FAR_FRR, auc_MTL, FAR_score, FRR_score, \
    evaluate_result_valid, evaluate_result_test, mkdir
from util.util_keras import set_trainability
from util.Utils import PrintScore1, ConfusionMatrix
from util.util import label2uniqueID, split_test_as_valid, argmin_mean_FAR_FRR, auc_MTL, FAR_score, FRR_score, \
    evaluate_result_valid, evaluate_result_test, mkdir, split_train_test
from util.util_keras import set_trainability, my_get_shape, outer_product, prob2extreme
from util.util import my_zscore_test, my_zscore
from sklearn.semi_supervised import LabelPropagation
from losses.losses import *
from tensorflow import set_random_seed
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn import svm

if K.backend() == "tensorflow":
    os.environ["CUDA_VISIBLE_DEVICES"] = '0,1,2'
    config = K.tf.ConfigProto()
    config.gpu_options.allow_growth = True
    session = K.tf.Session(config=config)
    K.set_session(session)


def readData(data_txt, SUB):
    data, label, subject = [], [], []
    with open(data_txt) as f:
        sub = 0
        lines = f.readlines()
        for l in lines:
            sub += 1
            if sub not in SUB:
                continue
            dataIn = loadmat(l.split()[0], mdict=None)
            tempData = np.asarray(dataIn["Fea"], dtype=np.float32)
            tempLabel = np.asarray(dataIn["Label"], dtype=np.float32)
            #tempLabel = tempLabel[0,:]

            tempSub = sub * np.ones(tempLabel.shape[1])
            if not len(data):
                data, label, subject = tempData, tempLabel, tempSub
            else:
                data = np.concatenate((data, tempData), axis=2)
                label = np.concatenate((label, tempLabel), axis=1)
                subject = np.concatenate((subject, tempSub), axis=0)
       # label = label.reshape(label.shape[1])
    return data, label, subject


# 第一阶段训练模型主函数

all_sub_rfc = []
all_sub_knn = []
all_sub_svm = []
start = time.time()
txtName = '/hdd/changhongli/Anxiety-AAL/LSTM/LOSO128_LSTM/svm_rfc_knn.txt'
for n in range(53):
    SUB = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28,
           29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53]

    print(SUB[n])
    dataDir = '/hdd/changhongli/Anxiety-AAL/dataDir.txt'

    data, label, subject = readData(dataDir, SUB)

    print('-----read data end-----')

    M = DataGenerate(data=data, label=label, subject=subject, testSub=SUB[n])
    train_data, train_label, test_data, test_label, train_y_c, test_y_c, ATTR_NUM, CLASS_NUM, input_shape, lambda_mat, prior_list = gen_raw_data(M)
    seed = 10
    # print(test_X.shape)
    test_label=test_label[:, 0]
    train_label = train_label[:, 0]
    idx = np.random.permutation(len(train_label))
    train_data = train_data[idx, :, :]
    train_label = train_label[idx]

    train_data = train_data.reshape(train_data.shape[0], train_data.shape[1] * train_data.shape[2])
    test_data = test_data.reshape(test_data.shape[0], test_data.shape[1] * test_data.shape[2])
    print(train_data.shape)
    # print(train_data.type)
    print(train_label.shape)
    rfc = RandomForestClassifier()
    rfc = rfc.fit(train_data, train_label)
    rfc_result = rfc.score(test_data, test_label)

    knn = KNeighborsClassifier()
    knn = knn.fit(train_data, train_label)
    knn_result = knn.score(test_data, test_label)

    SVM = svm.SVC()
    SVM = SVM.fit(train_data, train_label)
    svm_result = SVM.score(test_data, test_label)

    all_sub_rfc.append(rfc_result)
    all_sub_knn.append(knn_result)
    all_sub_svm.append(svm_result)

    # all_c_matrix.append(c_matrix)
    with open(txtName, 'a+') as t_f:
        t_f.write('\nsub = ' + str(subject) + ', rfc_result = ' + str(rfc_result) + ', knn_result = ' + str(
            knn_result) + ', svm_result = ' + str(svm_result))
        # t_f.write('\nconfusion matrix is:\n' + str(c_matrix))
        t_f.write('\n***********************************************************')

end = time.time()
# all_matrix = sum(all_c_matrix)

rfc_acc_mean = round(np.mean(all_sub_rfc), 4) * 100
rfc_acc_std = round(np.std(all_sub_rfc), 4) * 100

knn_acc_mean = round(np.mean(all_sub_knn), 4) * 100
knn_acc_std = round(np.std(all_sub_knn), 4) * 100

svm_acc_mean = round(np.mean(all_sub_svm), 4) * 100
svm_acc_std = round(np.std(all_sub_svm), 4) * 100

print('***********************************************************')
print('time is {}'.format(time.strftime('%Y%m%d_%H:%M:%S', time.localtime())))
print('rfc_mean/std = {}/{}, time is {}'.format(rfc_acc_mean, rfc_acc_std,
                                                time.strftime('%Y%m%d_%H:%M:%S', time.localtime())))
print('knn_mean/std = {}/{}'.format(knn_acc_mean, knn_acc_std))
print('svm_mean/std = {}/{}'.format(svm_acc_mean, svm_acc_std))

with open(txtName, 'a+') as t_f:
    t_f.write('\n\ntime is: ' + time.strftime('%Y%m%d_%H:%M:%S', time.localtime()))
    # t_f.write('\n\nconfusion matrix is:\n' + str(all_matrix))
    t_f.write('\nmean/std acc = %f/%f' % (rfc_acc_mean, rfc_acc_std))
    t_f.write('\nmean/std acc = %f/%f' % (knn_acc_mean, knn_acc_std))
    t_f.write('\nmean/std acc = %f/%f' % (svm_acc_mean, svm_acc_std))