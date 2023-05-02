# -*- coding: utf-8 -*-

import numpy as np
import h5py
import os.path
import numpy as np
import h5py
import os.path
from keras.models import load_model
import pdb
from scipy.io import savemat
from util.Utils import PrintScore, ConfusionMatrix
import pdb
from scipy.io import savemat

     
f = h5py.File('/redhdd/changhongli/Anxiety-AAL/results/pro3zuizhong/01/32_1554_data_phase1_acc_0.6354_0_0.9917_1_0.0083_2_0.0000.h5')


model=load_model('/redhdd/changhongli/Anxiety-AAL/results/pro3zuizhong/01/GAN_phase_1_task_1554_acc_0.6354.h5')
#f3 = h5py.File('/hdd/changhongli/pre/AAL-GCN-SEED/results/phase2-1/02/Result_GAN_phase_2_seen_outer_0_i_step_6_acc_0.8338.h5','r')
model.summary()

ATTR_NUM = int(np.array(f['ATTR_NUM']))
CLASS_NUM = np.array(f['CLASS_NUM']).astype(int)
train_f = np.array(f['train_f'])
#test_f = np.array(f['test_f'])
val_f = np.array(f['val_f'])
train_prob_set = [np.array(f['train_prob_%d' % i]) for i in range(ATTR_NUM)]
test_prob_set = [np.array(f['test_prob_%d' % i]) for i in range(ATTR_NUM)]
val_prob_set = [np.array(f['val_prob_%d' % i]) for i in range(ATTR_NUM)]
train_y_a = np.array(f['train_y_a'])
#test_y_a = np.array(f['test_y_a'])
val_y_a = np.array(f['val_y_a'])
train_y_c = np.array(f['train_y_c'])
#test_y_c = np.array(f['test_y_c'])
val_y_c = np.array(f['val_y_c'])
#hidden_dim = int(np.array(f['hidden_dim']))
feature_dim = int(np.array(f['feature_dim']))


print(feature_dim)
print(ATTR_NUM)



for i_prob, prob in enumerate(test_prob_set):
    prob = np.argmax(prob, axis=1)
    acc = PrintScore(val_y_a[:, i_prob], prob, savePath=save_root_path)
    ConfusionMatrix(val_y_a[:, i_prob], prob, classes=['Normal', 'Depression'],
                    savePath=save_root_path)
    print('acc:', acc)
    savemat('/redhdd/changhongli/Anxiety-AAL/results/pro3zuizhong/01/pro3_01.mat', {'prob': prob, 'val_prob_set': val_y_a[:, i_prob],'train_f':train_f, 'val_f':val_f, 'train_y_a':train_y_a,'val_y_a':val_y_a,'feature_dim':feature_dim})
    break


