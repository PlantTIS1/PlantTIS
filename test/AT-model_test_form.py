# -*- coding: utf-8 -*-
import sys
import numpy as np
import h5py
import scipy.io
import tensorflow as tf
tf.compat.v1.experimental.output_all_intermediates(True)
from keras.optimizers import RMSprop
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution1D, MaxPooling1D
#from keras.regularizers import l2, activity_l1
from keras.constraints import maxnorm
from keras.layers.recurrent import LSTM, GRU
from keras.models import load_model

def seq_matrix(seq_list,label):
  tensor = np.zeros((len(seq_list),203,8))
  for i in range(len(seq_list)):
    seq = seq_list[i]
    j = 0
    for s in seq:
      if s == 'A' and (j<100 or j>102):
        tensor[i][j] = [1,0,0,0,0,0,0,0]
      if s == 'T' and (j<100 or j>102):
        tensor[i][j] = [0,1,0,0,0,0,0,0]
      if s == 'C' and (j<100 or j>102):
        tensor[i][j] = [0,0,1,0,0,0,0,0]
      if s == 'G' and (j<100 or j>102):
        tensor[i][j] = [0,0,0,1,0,0,0,0]
      if s == '$':
        tensor[i][j] = [0,0,0,0,0,0,0,0]
      if s == 'A' and (j>=100 and j<=102):
        tensor[i][j] = [0,0,0,0,1,0,0,0]
      if s == 'T' and (j>=100 and j<=102):
        tensor[i][j] = [0,0,0,0,0,1,0,0]
      if s == 'C' and (j>=100 and j<=102):
        tensor[i][j] = [0,0,0,0,0,0,1,0]
      if s == 'G' and (j>=100 and j<=102):
        tensor[i][j] = [0,0,0,0,0,0,0,1]
      j += 1
  if label == 1:
    y = np.ones((len(seq_list),1))
  else:
    y = np.zeros((len(seq_list),1))
  return tensor, y

###### main function ######
print 'Loading test data...'

pos_seq_test = np.load(sys.argv[1], allow_pickle=True)
neg_seq_test = np.load(sys.argv[2], allow_pickle=True)

print len(pos_seq_test)
print(len(neg_seq_test))

pos_seq_test1 = []
neg_seq_test1 = []

for s in pos_seq_test:
    s1 = str(s[100:103]).replace("'", '').replace(' ', '').replace('[', '').replace(']', '')
    pos_seq_test1.append(s)

print str(len(pos_seq_test1))+' positive test data loaded...'
print (str(len(neg_seq_test1))+' negative test data loaded...')

pos_test_X, pos_test_y = seq_matrix(seq_list=pos_seq_test1, label=1)
neg_test_X, neg_test_y = seq_matrix(seq_list=neg_seq_test1, label=0)
X_test = pos_test_X
y_test = pos_test_y

print 'Loading model...'
print 'Predicting on test data...'
y_test_pred_n = np.zeros((len(y_test),1))
y_test_pred_p = np.zeros((len(y_test),1))


model_path = "/.../PlantTISTool/PlantTIS/train/model/"
k=[1,2,3,4,5,6,7,8,9,10]
for i in k:
  model = load_model(model_path + str(i) + "_model.hdf5") ###change
  y_test_pred = model.predict(X_test,verbose=1)
  y_test_pred_n += y_test_pred

y_test_pred_n = y_test_pred_n / 10

np.savetxt(sys.argv[3],y_test_pred_n,fmt='%.9f')

