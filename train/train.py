# coding:utf-8
import sys, os
import numpy as np
import h5py
import scipy.io
np.random.seed(1337) # for reproducibility

#import theano
import keras
from keras.preprocessing import sequence
from keras.optimizers import RMSprop
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution1D, MaxPooling1D
#from keras.regularizers import l2  # , activity_l1
from keras.constraints import maxnorm
from keras.layers.recurrent import LSTM, GRU
import matplotlib.pyplot as plt
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras import callbacks
from keras.callbacks import ReduceLROnPlateau

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


def process_seq(pos_seq_train, neg_seq_train):
	pos_seq_train1 = []
	neg_seq_train1 = []
	for s in pos_seq_train:
		pos_seq_train1.append(s)
	for s in neg_seq_train:
		neg_seq_train1.append(s)

	print(str(len(pos_seq_train1))+' positive test data loaded...')
	print(str(len(neg_seq_train1))+' negative test data loaded...')

	pos_train_X, pos_train_y = seq_matrix(seq_list=pos_seq_train1, label=1)
	neg_train_X, neg_train_y = seq_matrix(seq_list=neg_seq_train1, label=0)
	X_train = np.concatenate((pos_train_X,neg_train_X), axis=0)
	y_train = np.concatenate((pos_train_y,neg_train_y), axis=0)
	return X_train, y_train

print 'Loading test data...'
pos_seq_train = np.load('/home/malab8/Desktop/DeepTIS-main/train/data/train_p_set.npy',  allow_pickle=True)

for i in range(1, 11):
    neg_seq_train = np.load("/home/malab8/Desktop/DeepTIS-main/train/data/neg_train_" + str(i) + ".npy",  allow_pickle=True)

    pos_seq_valid = np.load('/home/malab8/Desktop/DeepTIS-main/train/data/val_p_set.npy',  allow_pickle=True)
    neg_seq_valid = np.load('/home/malab8/Desktop/DeepTIS-main/train/data/val_n_set_same.npy',  allow_pickle=True)

    X_train, y_train = process_seq(pos_seq_train = pos_seq_train, neg_seq_train = neg_seq_train)
    X_validation, y_validation = process_seq(pos_seq_train = pos_seq_valid, neg_seq_train = neg_seq_valid)

    print 'Building model...'
    model = Sequential()
    model.add(Convolution1D(nb_filter=128,
                        filter_length=3,
                        input_dim=8,
                        input_length=203,
                        border_mode='valid',
                        W_constraint = maxnorm(3),
                        activation='relu',
                        subsample_length=1))
    model.add(MaxPooling1D(pool_length=3))
    model.add(Dropout(p=0.21370950078747658))
    model.add(LSTM(output_dim=256,
               return_sequences=True))
    model.add(Dropout(p=0.7238091317104384))
    model.add(Flatten())
    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    class LossHistory(keras.callbacks.Callback):
        def on_train_begin(self, logs={}):
            self.losses = {'batch': [], 'epoch': []}
            self.accuracy = {'batch': [], 'epoch': []}
            self.val_loss = {'batch': [], 'epoch': []}
            self.val_acc = {'batch': [], 'epoch': []}

        def on_batch_end(self, batch, logs={}):
            self.losses['batch'].append(logs.get('loss'))
            self.accuracy['batch'].append(logs.get('acc'))
            self.val_loss['batch'].append(logs.get('val_loss'))
            self.val_acc['batch'].append(logs.get('val_acc'))

        def on_epoch_end(self, batch, logs={}):
            self.losses['epoch'].append(logs.get('loss'))
            self.accuracy['epoch'].append(logs.get('acc'))
            self.val_loss['epoch'].append(logs.get('val_loss'))
            self.val_acc['epoch'].append(logs.get('val_acc'))

        def loss_plot(self, loss_type,i):
            iters = range(len(self.losses[loss_type]))
            plt.figure()
            # acc
            plt.plot(iters, self.accuracy[loss_type], 'r', label='train acc')
            # loss
            plt.plot(iters, self.losses[loss_type], 'g', label='train loss')
            if loss_type == 'epoch':
                # val_acc
                plt.plot(iters, self.val_acc[loss_type], 'b', label='val acc')
                # val_loss
                plt.plot(iters, self.val_loss[loss_type], 'k', label='val loss')
            plt.grid(True)
            plt.xlabel(loss_type)
            plt.ylabel('acc-loss')
            plt.legend(loc="upper right")

            figure = plt.savefig(str(i) + ".png")
            return figure
    print 'Compiling model...'
    model.compile(loss='binary_crossentropy',
              optimizer='nadam',
              metrics=['accuracy'])
    early_stopping = EarlyStopping(monitor='val_loss',patience=5)
    modelDic = "/home/malab8/Desktop/lylbishe/apa/6/"
    history = LossHistory()

    filepath = str(i) + '_model.h5'
    checkpoint = ModelCheckpoint(os.path.join(modelDic, filepath), monitor='val_loss', verbose=1, save_best_only=True)

    history_callback = model.fit(x = X_train, y = y_train,
    validation_data=(X_validation, y_validation),
	batch_size = 64, epochs = 50,
	callbacks=[early_stopping, checkpoint, history])

    val_loss = np.array(history_callback.history["val_loss"])
    val_loss = val_loss[:,np.newaxis]
    train_loss = np.array(history_callback.history["loss"])
    train_loss = train_loss[:,np.newaxis]
    res_loss = np.concatenate((val_loss, train_loss), axis = 1)

    np.savetxt(fname = modelDic + "/" + str(i) + "-tracking-loss.txt", X = res_loss,
           delimiter = "\t", header = "\t".join(["val_loss","train_loss"]))

    history.loss_plot(loss_type='epoch',i="i")


