# -*- coding: utf-8 -*-

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

import numpy as np
import matplotlib.pyplot as plt
from itertools import cycle

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

from nets.nets import input_model_lstm_deep as input_model

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


def make_gan(inputs, G, D, G_trainable, D_trainable):
    set_trainability(G, G_trainable)
    set_trainability(D, D_trainable)
    x = G(inputs)
    output = D(x)
    return output


def make_gan_phase_1_gen_hidden_feature(inputs, G_in, G_set):
    output_set = []
    ATTR_NUM = len(G_set)
    for i in range(ATTR_NUM):
        x = G_set[i](G_in)
        output_set.append(x)
    feats = concatenate(output_set)
    GAN = Model(inputs, feats)
    return GAN, feats


def make_gan_phase_1_task(inputs, GAN_in, G_set, D_set, loss, opt, loss_weights):
    output_set = []
    ATTR_NUM = len(G_set)

    G_trainable = True
    D_trainable = True
    for i in range(ATTR_NUM):
        output_ii = make_gan(GAN_in, G_set[i], D_set[i][i], G_trainable, D_trainable)
        output_set.append(output_ii)
    model = Model(inputs, output_set)
    model.compile(loss=loss, optimizer=opt, loss_weights=loss_weights)
    return model, output_set


def make_gan_phase_1_domain_pos(inputs, GAN_in, G_set, D_set, loss, opt, loss_weights):
    output_set = []
    ATTR_NUM = len(G_set)
    G_trainable = False
    D_trainable = True
    for i in range(ATTR_NUM):
        for j in range(ATTR_NUM):
            if i == j:
                D_trainable = False
            else:
                D_trainable = True
            output_ij = make_gan(GAN_in, G_set[i], D_set[i][j], G_trainable, D_trainable)
            output_set.append(output_ij)
    GAN = Model(inputs, output_set)
    GAN.compile(loss=loss, optimizer=opt, loss_weights=loss_weights)
    return GAN, output_set


def make_gan_phase_1_domain_neg(inputs, GAN_in, G_set, D_set, loss, opt, loss_weights):
    output_set = []
    ATTR_NUM = len(G_set)
    D_trainable = False
    G_trainable = True
    for i in range(ATTR_NUM):
        for j in range(ATTR_NUM):
            if i == j:
                G_trainable = False
            else:
                G_trainable = True
            output_ij = make_gan(GAN_in, G_set[i], D_set[i][j], G_trainable, D_trainable)
            output_set.append(output_ij)
    GAN = Model(inputs, output_set)
    GAN.compile(loss=loss, optimizer=opt, loss_weights=loss_weights)
    return GAN, output_set


def build_model(ATTR_NUM, CLASS_NUM, feature_dim, hidden_dim, input_shape, lambda_mat):
    model_input, inputs, _, shared_dim = input_model(hidden_dim, input_shape, depth=5)

    G_set_phase_1 = []
    D_set_phase_1 = []
    for i in range(ATTR_NUM):
        G, _ = get_generative(input_dim=shared_dim, out_dim=feature_dim)
        G_set_phase_1.append(G)
        D_set_sub = []
        for j in range(ATTR_NUM):
            if i == j:
                activation = 'softmax'
            else:
                activation = 'softmax'
            D, _ = get_discriminative(input_dim=feature_dim, out_dim=CLASS_NUM[j], activation=activation)
            D_set_sub.append(D)
        D_set_phase_1.append(D_set_sub)

    opt_gan = Adam(lr=0.0002, beta_1=0.5, beta_2=0.999)
    opt = Adam(lr=1e-3)

    loss_weights = [1.]
    loss_weights.extend([0.1 for _ in range(ATTR_NUM - 1)])

    set_trainability(model_input, True)

    feats = model_input(inputs)
    loss = [K.categorical_crossentropy for _ in range(ATTR_NUM)]
    GAN_phase_1_task, _ = make_gan_phase_1_task(inputs, feats, G_set_phase_1, D_set_phase_1, loss, opt, loss_weights)

    for i in range(ATTR_NUM):
        loss_weights = [1.]
        loss_weights.extend([0.1 for _ in range(ATTR_NUM - 1)])
        for j in range(ATTR_NUM):
            if i != j:
                loss_weights[j] = loss_weights[j] * lambda_mat[j, i]
        if i == 0:
            loss_w = loss_weights
        else:
            loss_w.extend(loss_weights)
    loss_weights = loss_w

    set_trainability(model_input, False)
    feats = model_input(inputs)
    GAN_phase_1_domain_pos, _ = make_gan_phase_1_domain_pos(inputs, feats, G_set_phase_1, D_set_phase_1, 'categorical_crossentropy', opt_gan,
                                                            loss_weights)
    set_trainability(model_input, True)
    feats = model_input(inputs)
    GAN_phase_1_domain_neg, _ = make_gan_phase_1_domain_neg(inputs, feats, G_set_phase_1, D_set_phase_1, 'categorical_crossentropy', opt_gan,
                                                            loss_weights)

    Model_gen_hidden_feature, _ = make_gan_phase_1_gen_hidden_feature(inputs, feats, G_set_phase_1)

    return GAN_phase_1_task, GAN_phase_1_domain_pos, GAN_phase_1_domain_neg, Model_gen_hidden_feature

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

def train(save_root_path):
    all_sub_acc = []

    Test_Y = []
    Probs = []
    start = time.time()
    Val_f = np.zeros((480, 256))

    feature_dim = 128
    batch_size = 32
    n_step = 50
    hidden_dim = 64
    for n in range(53):
        SUB = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28,
               29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53]

        print(SUB[n])
        dataDir = '/hdd/changhongli/Anxiety-AAL/dataDir3.txt'

        data, label, subject = readData(dataDir, SUB)

        print('-----read data end-----')

        M = DataGenerate(data=data, label=label, subject=subject, testSub=SUB[n])

        train_X, train_y_a, test_X, test_y_a, train_y_c, test_y_c, ATTR_NUM, CLASS_NUM, input_shape, lambda_mat, prior_list = gen_raw_data(M)

        idx_valid, idx_test = split_test_as_valid(train_y_c)
        val_X = train_X[idx_valid]
        val_y_a = train_y_a[idx_valid]
        val_y_c = train_y_c[idx_valid]

        train_X = train_X[idx_test]
        train_y_a = train_y_a[idx_test]
        train_y_c = train_y_c[idx_test]


        mkdir(save_root_path)
        write_path1 = '%s_data_phase1.h5' % (SUB[n])
        write_path2 = '%s_GAN_phase_1_task.h5' % (SUB[n])
        write_path3 = '%s_Model_gen_hidden_feature.h5' % (SUB[n])


        print(write_path2)
        model=load_model(os.path.join(save_root_path, write_path2))
        probs = model.predict(val_X)
        for i_prob, prob in enumerate(probs):
            prob = np.argmax(prob, axis=1)
            test_y = val_y_a[:, i_prob]
            acc_max = np.mean(prob == test_y)
            break
        test_Y = test_y
        print("Classification accuracy: %f " % (acc_max))
        print(write_path3)
        model1=load_model(os.path.join(save_root_path, write_path3))
        val_f = model1.predict(test_X)

        #pdb.set_trace()
        Val_f = np.concatenate((Val_f, val_f), axis=0)
        #Val_f.append(val_f)
        all_sub_acc.append(acc_max)

        Test_Y = np.concatenate((Test_Y, test_Y), axis=0)
        Probs = np.concatenate((Probs, prob), axis=0)
    end = time.time()
    acc_mean = round(sum(all_sub_acc) / 53, 4) * 100
    acc_std = round(np.std(all_sub_acc), 4) * 100
    PrintScore1(Test_Y, Probs, savePath=save_root_path)
    ConfusionMatrix(Test_Y, Probs, classes=['Normal', 'Depression'],
                    savePath=save_root_path)

    savemat('/hdd/changhongli/Anxiety-AAL/results/LOSO_lanzhou3/lanzhou3_LOSO.mat', {'Val_f ': Val_f, 'all_sub_acc': all_sub_acc,'Test_Y':Test_Y, 'Probs':Probs, 'acc_mean':acc_mean,'acc_std':acc_std})
    return

if __name__ == "__main__":

    import os, signal, traceback

    try:
        save_root_path = '/hdd/changhongli/Anxiety-AAL/results/LOSO_lanzhou3/'
        train(save_root_path)
    except:
        traceback.print_exc()
    finally:
        os.kill(os.getpid(), signal.SIGKILL)