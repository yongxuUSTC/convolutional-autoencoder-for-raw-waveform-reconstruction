 
import sys
sys.path.append('/user/HS103/yx0001/Downloads/Hat')
import pickle
import numpy as np
np.random.seed(1515)
import scipy.stats

import keras
from keras.models import load_model

import config as cfg
import prepare_data_raw_1ch as pp_data
import csv
from Hat.preprocessing import reshape_3d_to_4d
from Hat.preprocessing import sparse_to_categorical, mat_2d_to_3d
from Hat.preprocessing import sparse_to_categorical, mat_2d_to_3d, reshape_3d_to_4d, mat_concate_multiinmaps4in
from Hat.metrics import prec_recall_fvalue
import cPickle
from keras import backend as K
import wavio
import matplotlib.pyplot as plt
np.set_printoptions(threshold=np.nan, linewidth=1000, precision=2, suppress=True)

import scipy
from scipy.signal import freqz,freqs

def reconstructWav( X ):
    N = len(X)
    X=np.array(X)
    Y=np.array(X[0,0:512/2]) # for the first half, did not include the last num
    for i in range(1,N-1):
        temp1=(X[i-1,512/2:512]+X[i,0:512/2])/2
        Y=np.concatenate((Y,temp1),axis=0) # for the middle halfs
    temp2=X[N-1,512/2:512]
    Y=np.concatenate((Y,temp2),axis=0) # for the last half
    return Y

def reshapeX( X ):
    N = len(X)
    return X.reshape( (N, 1, 1, feadim) )

def reshapeX2( X ):
    N = len(X)
    return X.reshape( (N, 257) )

def reshapeX3( X ):
    N = len(X)
    return X.reshape( (N, feadim) )

def reshapeX4( X ):
    N = len(X)
    return X.reshape( (N, 256) )

debug=0
feadim=512
# hyper-params
n_labels = len( cfg.labels )
fe_fd = cfg.dev_fe_mel_fd
#fe_fd_ori = cfg.dev_fe_mel_fd_ori
agg_num = 1        # concatenate frames
hop = 1          # step_len
n_hid = 1000
fold = 9  # can be 0, 1, 2, 3, 4

# load model
dae=load_model('/vol/vssp/msos/yx/chime_home/DCASE2016_task4_scrap_2ch_wav_ipd_ild_overlap/Md/deconvAE_keras_kernelsize5_weights.100-0.00.hdf5')
#dae=load_model('/vol/vssp/msos/yx/chime_home/DCASE2016_task4_scrap_2ch_wav_ipd_ild_overlap/Md/deconvAE_keras_weights.22-0.00.hdf5-fold1-convDeconvAE')

def recognize():
    
    
    with open( cfg.dev_cv_csv_path, 'rb') as f:
        reader = csv.reader(f)
        lis = list(reader)
        nl=0
        # read one line
        for li in lis:
            na = li[1]
            curr_fold = int(li[2])
            
            nl=nl+1
            #if fold==curr_fold:           
            if 1==1:
                fe_path = fe_fd + '/' + na + '.f'
                print na
                print nl
                X = cPickle.load( open( fe_path, 'rb' ) )

                #X = scaler.transform( X )
                #print X.shape
                X3d = mat_2d_to_3d( X, agg_num, hop )
                #print X3d.shape
                X3d= reshapeX(X3d)
                #print X3d.shape

                if debug:
                    pred = dae.predict( X3d )
                    pred=reshapeX3( pred )
                    print pred.shape

                get_3rd_layer_output = K.function([dae.layers[0].input, K.learning_phase()], [dae.layers[6].output])
                layer_output = get_3rd_layer_output([X3d, 0])[0]
                layer_output = reshapeX4(layer_output)
                print layer_output.shape

                out_path = '/vol/vssp/msos/yx/chime_home/DCASE2016_task4_scrap_2ch_wav_ipd_ild_overlap/Fe/convAE_m' + '/' + na + '.f' #### change na[0:-4]
                if not debug:
                    cPickle.dump( layer_output, open(out_path, 'wb'), protocol=cPickle.HIGHEST_PROTOCOL )
                    print 'write done!'              
                #sys.exit()
                if debug:
                    print layer_output.shape
                    #layer_output1=layer_output[5,:]
                    layer_output=reshapeX4(layer_output)
                    print layer_output.shape
                    imgplot=plt.matshow((layer_output[:,:]))
                    plt.colorbar()
                    plt.show()
                    #sys.pause()
                if debug:
                #if nl==1:
                    #fig=plt.figure()
                    #ax=fig.add_subplot(2,1,1)
                    #nfr=3
                    #plt.plot(np.ravel(pred))
                    #wavio.write('out.wav',np.ravel(pred),16000,sampwidth=2)
                    wavio.write('out.wav',reconstructWav(pred),16000,sampwidth=2)
                    
                    print np.max(pred),np.min(pred)
                    #ax.matshow(pred[:,nfr*fea_dim:(nfr+1)*fea_dim].T, origin='lower', aspect='auto')
                    #ax=fig.add_subplot(2,1,2)
                    X3d=reshapeX3(X3d)
                    #plt.plot(np.ravel(X3d))
                    #wavio.write('in.wav',np.ravel(X3d),16000,sampwidth=2)
                    wavio.write('in.wav',reconstructWav(X3d),16000,sampwidth=2)
                    print np.max(X3d),np.min(X3d)
                    #plt.show()
                    pause
                if debug:
                    w1=dae.layers[1].get_weights()[0]
                    print w1.shape
                    b=w1[150,0,0,:]
                    print b.shape
                    #plt.axis('off')
                    #b=scipy.signal.firwin(40,0.5,window=('kaiser',8))
                    #print b
                    plt.subplot(2,1,1)
                    plt.plot(b,'b')
                    plt.subplot(2,1,2)
                    w,h=freqz(b)
                    m=20*np.log10(np.divide(abs(h),max(abs(h))))
                    mi=np.argmin(m)
                    print mi
                    ma=np.argmax(m)
                    print ma
                    plt.plot(w,m,'r')
                    plt.show()
                    pause

if __name__ == '__main__':
    recognize()
