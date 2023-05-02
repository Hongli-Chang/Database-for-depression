# -*- coding: utf-8 -*-

import numpy as np
import h5py
import os.path

import pdb
from scipy.io import savemat
#from util.Utils import PrintScore, ConfusionMatrix

f = h5py.File('/redhdd/changhongli/Anxiety-AAL/results/test_resuits/data_phaselanzhou.h5')

ACC = np.array(f['ACC'])
Task_Loss = np.array(f['Task_Loss'])
Pos_Loss = np.array(f['Pos_Loss'])
Neg_Loss = np.array(f['Neg_Loss'])

savemat('data_phaselanzhou.mat', {'ACC': ACC, 'Task_Loss': Task_Loss,'Pos_Loss': Pos_Loss,'Neg_Loss': Neg_Loss})




