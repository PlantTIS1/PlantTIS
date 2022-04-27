# -*- coding: utf-8 -*-
import sys
import os
import re
import numpy as np
import h5py
import scipy.io
import tensorflow as tf
#tf.compat.v1.experimental.output_all_intermediates(True)
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

pos_seq_test = np.load('test_p_set.npy', allow_pickle=True)
neg_seq_test = np.load('neg_test.npy', allow_pickle=True)

print (len(pos_seq_test))
print (len(neg_seq_test))

pos_seq_test1 = []
neg_seq_test1 = []

for s in pos_seq_test:
    s1 = str(s[100:103]).replace("'", '').replace(' ', '').replace('[', '').replace(']', '')
    pos_seq_test1.append(s)
for s in neg_seq_test:
    s1 = str(s[100:103]).replace("'", '').replace(' ', '').replace('[', '').replace(']', '')
    neg_seq_test1.append(s)
print (str(len(pos_seq_test1))+' positive test data loaded...')
print (str(len(neg_seq_test1))+' negative test data loaded...')

pos_test_X, pos_test_y = seq_matrix(seq_list=pos_seq_test1, label=1)
neg_test_X, neg_test_y = seq_matrix(seq_list=neg_seq_test1, label=0)
'''''''''
X_test = pos_test_X
y_test = pos_test_y
'''
X_test = np.concatenate((pos_test_X, neg_test_X), axis=0)
y_test = np.concatenate((pos_test_y, neg_test_y), axis=0)

print 'Loading model...'
print 'Predicting on test data...'
y_test_pred_n = np.zeros((len(y_test),1))
y_test_pred_p = np.zeros((len(y_test),1))

#model_path = sys.argv[3]
model_path='/home/malab8/Desktop/lylbishe/apa/6/'
''''
nums = []
for root, dirs, files in os.walk(model, topdown=False):
    for name in files:
      #print name
      num = re.search(r"\d*_", name).group().replace("_", "")
      nums.append(num)
      print num
    for name in dirs:
      pass
print nums

for i in range(1, len(nums)+1):
  print i
  model = load_model(str(i) + "_model.h5")  ###change
  y_test_pred = model.predict(X_test, verbose=1)
  y_test_pred_n += y_test_pred
y_test_pred_n = y_test_pred_n / 10
'''
k=[1,2,3,4,5,6,7,8,9,10]
for i in k:
  model = load_model(model_path+str(i)+'_model'+'.h5') ###change
  y_test_pred = model.predict(X_test,verbose=1)
  y_test_pred_n += y_test_pred


y_test_pred_n = y_test_pred_n / 10

#np.savetxt('zea_lyl.txt', y_test_pred_n, fmt='%.9f')

#yuzhi# 阈值
from sklearn.metrics import precision_recall_curve
import numpy as np

precisions, recalls, thresholds = precision_recall_curve(y_test,y_test_pred_n)

# 拿到最优结果以及索引
f1_scores = (2 * precisions * recalls) / (precisions + recalls)
best_f1_score = np.max(f1_scores[np.isfinite(f1_scores)])
best_f1_score_index = np.argmax(f1_scores[np.isfinite(f1_scores)])
'''''''''''
import matplotlib.pyplot as plt
plt.plot(precisions, recalls)
plt.show()
# 阈值
print best_f1_score, thresholds[best_f1_score_index]


'''

#pinggu
import matplotlib.pyplot as plt
from sklearn.metrics import average_precision_score, roc_auc_score,roc_curve,auc
fpr,tpr,threshold = roc_curve(y_test,y_test_pred_n ) #为什么要用y_test
roc_auc = auc(fpr,tpr)

plt.figure()
lw = 2
plt.figure(figsize=(10,10))
plt.plot(fpr, tpr, color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()

print 'Perf without prior, AUC: '+str(roc_auc_score(y_test, y_test_pred_n))
print 'Perf without prior, AUPR: '+str(average_precision_score(y_test, y_test_pred_n))
'''''''''
print 'Perf with prior, AUC: '+str(roc_auc_score(y_test, y_test_pred_p))
print 'Perf with prior, AUPR: '+str(average_precision_score(y_test, y_test_pred_p))

'''