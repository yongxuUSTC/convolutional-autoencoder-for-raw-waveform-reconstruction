import sys
sys.path.append('/user/HS103/yx0001/Downloads/Hat')
import pickle
import numpy as np
np.random.seed(1515)
import os
import config as cfg
from Hat.preprocessing import reshape_3d_to_4d
import prepare_data_raw_1ch as pp_data
#from prepare_data import load_data


import keras

from keras.datasets import mnist, cifar10
from keras.models import Sequential,Model
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import ZeroPadding2D, AveragePooling2D, Convolution2D, Deconvolution2D,MaxPooling2D, Convolution1D,MaxPooling1D,UpSampling2D
from keras.utils import np_utils
from keras.layers import Merge, UpSampling2D, Input
from keras.regularizers import l1, l2, activity_l2
from keras.callbacks import ModelCheckpoint
from keras.layers.normalization import BatchNormalization
from keras.layers import LSTM, SimpleRNN, GRU, TimeDistributed, Bidirectional, Permute
import h5py
from keras.optimizers import SGD,Adam

# resize data for fit into CNN. size: (batch_num*color_maps*height*weight)
def reshapeX( X ):
    N = len(X)
    return X.reshape( (N, 1, 1, feadim) )


feadim=512

# hyper-params
fe_fd = cfg.dev_fe_mel_fd

#fe_fd_ori = cfg.dev_fe_mel_fd_ori
agg_num = 1        # concatenate frames
hop = 1            # step_len
n_hid = 1000
n_out = len( cfg.labels )
print n_out
fold = 9           # can be 0, 1, 2, 3, 4

# prepare data
tr_X, tr_y, te_X, te_y = pp_data.GetAllData( fe_fd, agg_num, hop, fold )
[batch_num, n_time, n_freq] = tr_X.shape
#print tr_X.shape #(188000, 1, 512)   
print tr_X.shape, tr_y.shape
print te_X.shape, te_y.shape

tr_X,te_X=reshapeX(tr_X),reshapeX(te_X)
print tr_X.shape, tr_y.shape
print te_X.shape, te_y.shape

###build model by keras

alpha=1
kernelsize=5

input_audio=Input(shape=(1,1,512))

x = Convolution2D(alpha*64,1,kernelsize,activation='relu',border_mode='same')(input_audio)
x = AveragePooling2D((1,2))(x)
x = Convolution2D(alpha*32,1,kernelsize,activation='relu',border_mode='same')(x)
x = AveragePooling2D((1,2))(x)
x = Convolution2D(alpha*4,1,kernelsize,activation='relu',border_mode='same')(x)
encoded = AveragePooling2D((1,2))(x)

x = Deconvolution2D(alpha*4,1,kernelsize,output_shape=(None,4,1,64),activation='relu',border_mode='same')(encoded)
#x = Convolution2D(alpha*4,1,kernelsize,activation='relu',border_mode='same')(encoded)
x = UpSampling2D((1,2))(x)
x = Deconvolution2D(alpha*32,1,kernelsize,output_shape=(None,32,1,128),activation='relu',border_mode='same')(x)
#x = Convolution2D(alpha*32,1,kernelsize,activation='relu',border_mode='same')(x)
x = UpSampling2D((1,2))(x)
x = Deconvolution2D(alpha*64,1,kernelsize,output_shape=(None,64,1,256),activation='relu',border_mode='same')(x)
#x = Convolution2D(alpha*64,1,kernelsize,activation='relu',border_mode='same')(x)
x = UpSampling2D((1,2))(x)
decoded = Deconvolution2D(1,1,kernelsize,output_shape=(None,1,1,512),activation='linear',border_mode='same')(x)
#decoded = Convolution2D(1,1,3,activation='linear',border_mode='same')(x)

autoencoder=Model(input_audio, decoded)
autoencoder.summary()


autoencoder.compile(optimizer='adam',loss='mse')


dump_fd=cfg.scrap_fd+'/Md/deconvAE_keras_kernelsize5_weights.{epoch:02d}-{val_loss:.2f}.hdf5'

eachmodel=ModelCheckpoint(dump_fd,monitor='val_loss',verbose=0,save_best_only=False,save_weights_only=False,mode='auto')    

autoencoder.fit(tr_X, tr_X, batch_size=100, nb_epoch=101,
              verbose=1, validation_data=(te_X, te_X), callbacks=[eachmodel]) #, callbacks=[best_model])

