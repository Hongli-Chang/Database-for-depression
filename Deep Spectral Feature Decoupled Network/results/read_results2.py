# -*- coding: utf-8 -*-

import numpy as np
import h5py
import os.path

import pdb
from scipy.io import savemat

     
f = h5py.File('/hdd/changhongli/pre/AAL-GCN-SEED/results/0134/01/32_7002_data_phase1_acc_0.9219.h5')
ATTR_NUM = int(np.array(f['ATTR_NUM']))
CLASS_NUM = np.array(f['CLASS_NUM']).astype(int)
train_f = np.array(f['train_f'])
#test_f = np.array(f['test_f'])
val_f = np.array(f['val_f'])
train_prob_set = [np.array(f['train_prob_%d' % i]) for i in range(ATTR_NUM)]
#test_prob_set = [np.array(f['test_prob_%d' % i]) for i in range(ATTR_NUM)]
val_prob_set = [np.array(f['val_prob_%d' % i]) for i in range(ATTR_NUM)]
train_y_a = np.array(f['train_y_a'])
#test_y_a = np.array(f['test_y_a'])
val_y_a = np.array(f['val_y_a'])
train_y_c = np.array(f['train_y_c'])
#test_y_c = np.array(f['test_y_c'])
val_y_c = np.array(f['val_y_c'])
#hidden_dim = int(np.array(f['hidden_dim']))
feature_dim = int(np.array(f['feature_dim']))

savemat('phase1_01.mat', {'train_f':train_f, 'val_f':val_f, 'train_y_a':train_y_a})




