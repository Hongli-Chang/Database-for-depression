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

from data.gendata_seed_grouped_classes_LSTM_lanzhou import gen_raw_data

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

all_sub_acc = []
all_c_matrix = []
Test_Y = []
Probs = []
start = time.time()
for n in range(53):
    SUB = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28,
           29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53]

    print(SUB[n])
    dataDir = '/hdd/changhongli/Anxiety-AAL/dataDir.txt'

    data, label, subject = readData(dataDir, SUB)

    print('-----read data end-----')

    M = DataGenerate(data=data, label=label, subject=subject, testSub=SUB[n])
    train_X, train_y_a, test_X, test_y_a, train_y_c, test_y_c, ATTR_NUM, CLASS_NUM, input_shape, lambda_mat, prior_list = gen_raw_data(M)
    test_y_a=test_y_a[:, 0]
    train_y_a = train_y_a[:, 0]
    train_y_a = np_utils.to_categorical(train_y_a, num_classes=2)
    test_y_a = np_utils.to_categorical(test_y_a, num_classes=2)
    seed = 1
    print(test_X.shape)
    idx = np.random.permutation(len(train_y_a))
    train_X = train_X[idx, :, :]
    train_y_a = train_y_a[idx]
    train_X, X_validate, train_y_a, Y_validate = train_test_split(train_X, train_y_a, test_size=0.3, random_state=seed)
    #print(test_y_a)
    #train_X, X_validate, train_Y, Y_validate, test_X, test_Y = M.train_X, M.X_validate, M.train_Y, M.Y_validate, M.test_data, M.test_label
    acc_max = 0
    print("--------------dataprocess end -----------------")
    hidden_dim = 64
    print(input_shape)
    model = input_model(hidden_dim, input_shape, depth=5)
    #
    # # compile the model and set the optimizers
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    WEIGHTS_PATH = r'/hdd/changhongli/Anxiety-AAL/LSTM/LOSO128_LSTM/%s.h5' % (SUB[n])
    checkpointer = ModelCheckpoint(filepath=WEIGHTS_PATH, verbose=1, save_best_only=True)
    #
    model.fit(train_X, train_y_a, batch_size=64, epochs=20, verbose=2, validation_data=(X_validate, Y_validate),
              shuffle=True, callbacks=[checkpointer])
    print(WEIGHTS_PATH)
    model.load_weights(WEIGHTS_PATH)
    probs = model.predict(test_X)
    preds = probs.argmax(axis=-1)
    test_y_a = test_y_a.argmax(axis=-1)
    acc_max = np.mean(preds == test_y_a)

    print("Classification accuracy: %f " % (acc_max))
    #c_matrix = confusion_matrix(test_y_a, preds)
    all_sub_acc.append(acc_max)
    #all_c_matrix.append(c_matrix)

    Test_Y = np.concatenate((Test_Y, test_y_a), axis=0)
    Probs = np.concatenate((Probs, preds), axis=0)

end = time.time()
acc_mean = round(sum(all_sub_acc) / 53, 4) * 100
acc_std = round(np.std(all_sub_acc), 4) * 100
#all_matrix = sum(all_c_matrix)
WEIGHTS_PATH1 = r'/hdd/changhongli/Anxiety-AAL/LSTM/LOSO128_LSTM/'
#acc = PrintScore(Test_Y, Probs, savePath=WEIGHTS_PATH1)
#ConfusionMatrix(Test_Y, Probs, classes=['Normal', 'Depression'], savePath=WEIGHTS_PATH1)
print('***********************************************************')
print(all_sub_acc)
print('time is {}'.format(time.strftime('%Y%m%d_%H:%M:%S', time.localtime())))
print('mean/std = {}/{}, time is {}'.format(acc_mean, acc_std, time.strftime('%Y%m%d_%H:%M:%S', time.localtime())))

txtName = '/hdd/changhongli/Anxiety-AAL/LSTM/LOSO128_LSTM/'
txtName += time.strftime('%Y%m%d_%H:%M:%S', time.localtime()) + '.txt'

with open(txtName, 'a+') as t_f:
    t_f.write('\n\ntime is: ' + time.strftime('%Y%m%d_%H:%M:%S', time.localtime()))
    #t_f.write('\n\nconfusion matrix is:\n' + str(all_matrix))
    t_f.write('\nmean/std acc = %.2f/%.2f' % (acc_mean, acc_std))
    t_f.write('\n\nall_sub_acc:\n' + str(all_sub_acc))
