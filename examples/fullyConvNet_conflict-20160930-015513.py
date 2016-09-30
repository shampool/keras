import sys
import os
os.environ['KERAS_BACKEND'] = 'theano'
#os.environ['THEANO_FLAGS'] = 'device=gpu0,optimizer=fast_run,force_device=True, allow_gc=True'
os.environ['THEANO_FLAGS'] = 'device=gpu0'
# you dont need to install this fork
keras_root = '..'
kerasversion = 'keras-1'
sys.path.insert(0, os.path.join(keras_root))
sys.path.insert(0, os.path.join(keras_root,'keras'))
sys.path.insert(0, os.path.join(keras_root,'keras','layers'))

from keras import backend as K
from keras.layers import Input,Dropout, merge,LocalResponseNorm,BatchNormalization, Dense,Lambda
from keras.layers.convolutional import  MaxPooling2D, UpSampling2D, Resize2D,Convolution2D # BLUpSampling2D,
from keras.optimizers import Adadelta  ,SGD, RMSprop, adam
from keras.layers.advanced_activations import ELU

from keras.regularizers import l2
from keras.models import Model
import numpy as np

def buildMixModel(img_channels=3, lr = 0.01,weight_decay = 1e-7, loss='mse',activ='relu', last_activ='sigmoid'):
    # just build a tiny fcn model, you can use more layers and more filters as you want

    main_input = Input(shape=(img_channels, None, None), name='input')
    conv_1 = Convolution2D(4,3,3, border_mode = 'same', activation= activ, init='orthogonal',name='conv_1',
                                 W_regularizer=l2(weight_decay), b_regularizer=l2(weight_decay))(main_input)
    
    max_1 = MaxPooling2D(pool_size = (2,2))(conv_1)

    conv_2 = Convolution2D(8,3,3, border_mode = 'same', activation=activ, init='orthogonal',name='conv_2',
                                 W_regularizer=l2(weight_decay), b_regularizer=l2(weight_decay))(max_1)
    max_2 = MaxPooling2D(pool_size = (2,2))(conv_2)
    dp_0 =  Dropout(0.25)(max_2) 
    conv_3 = Convolution2D(16,3,3, border_mode = 'same', activation= activ, init='orthogonal',name='conv_3',
                                 W_regularizer=l2(weight_decay), b_regularizer=l2(weight_decay))(dp_0)  # 25
    max_3 = MaxPooling2D(pool_size = (2,2))(conv_3)                                                      # 12

    conv_4 = Convolution2D(32,3,3, border_mode = 'same', activation=activ, init='orthogonal',name='conv_4',
                                 W_regularizer=l2(weight_decay), b_regularizer=l2(weight_decay))(max_3)  # 12
    max_4 = MaxPooling2D(pool_size = (2,2))(conv_4)                                                      # 12
    dp_1 =  Dropout(0.25)(max_4)
    conv_5 = Convolution2D(64,3,3, border_mode = 'same', activation=activ, init='orthogonal',name='conv_5',
                                 W_regularizer=l2(weight_decay), b_regularizer=l2(weight_decay))(dp_1)  # 6

    upsamp_0 = UpSampling2D((2,2))(conv_5)
    resize_0 = Resize2D(K.shape(conv_4))(upsamp_0)
    deconv_0 = Convolution2D(32,3,3, border_mode = 'same', activation=activ, init='orthogonal',name='deconv_0',
                                   W_regularizer=l2(weight_decay), b_regularizer=l2(weight_decay))(resize_0)
    dp_2 =  Dropout(0.25)(deconv_0)
    upsamp_1 = UpSampling2D((2,2))(dp_2)
    resize_1 = Resize2D(K.shape(conv_3))(upsamp_1)
    deconv_1 = Convolution2D(16,3,3, border_mode = 'same', activation=activ, init='orthogonal',name='deconv_1',
                                   W_regularizer=l2(weight_decay), b_regularizer=l2(weight_decay))(resize_1)

    upsamp_2 = UpSampling2D((2,2))(deconv_1)
    resize_2 = Resize2D(K.shape(conv_2))(upsamp_2)
    deconv_2 = Convolution2D(8,3,3, border_mode = 'same', activation=activ,init='orthogonal',name='deconv_2',
                                   W_regularizer=l2(weight_decay), b_regularizer=l2(weight_decay))(resize_2)

    dp_3 =  Dropout(0.25)(deconv_2)
    upsamp_3 = UpSampling2D((2,2))(dp_3)
    resize_3 = Resize2D(K.shape(conv_1))(upsamp_3)
    deconv_3 = Convolution2D(4,3,3, border_mode = 'same', activation=activ,init='orthogonal',name='deconv_3',
                                   W_regularizer=l2(weight_decay), b_regularizer=l2(weight_decay))(resize_3)


    last_conv = Convolution2D(1,3,3, border_mode = 'same', activation=last_activ,init='orthogonal', name= 'output_mask',
                                       W_regularizer=l2(weight_decay), b_regularizer=l2(weight_decay))(deconv_3)

    model = Model(input=[main_input], output=[last_conv])
    #opt = SGD(lr=lr, decay= 1e-6, momentum=0.9,nesterov=True)
    #opt = Adadelta(lr=lr, rho=0.95, epsilon=1e-06,clipvalue=10)
    opt = adam(lr=lr)
    model.compile(loss={'output_mask': loss }, optimizer=opt)
    return model

def Indxflow(Totalnum, batch_size):
    numberofchunk = int(Totalnum + batch_size - 1)// int(batch_size)   # the floor
    #Chunkfile = np.zeros((batch_size, row*col*channel))
    totalIndx = np.random.permutation(np.arange(Totalnum))

    chunkstart = 0
    for chunkidx in range(int(numberofchunk)):
        thisnum = min(batch_size, Totalnum - chunkidx*batch_size)
        thisInd = totalIndx[chunkstart: chunkstart + thisnum]
        chunkstart += thisnum
        yield thisInd


def batchflow(batch_size, *Data):
    # we dont check Data, they should all have equal first dimension
    Totalnum = Data[0].shape[0]
    for thisInd in Indxflow(Totalnum, batch_size):
        if len(Data) == 1:
            yield Data[0][thisInd, ...]
        else:
            batch_tuple = [s[thisInd,...] for s in Data]
            yield tuple(batch_tuple)


batch_size = 2
weightspath = 'weights.h5'
nsamples = 1000
input_dimension = (nsamples, 3, 224,224) # nsample * channel * row * col
output_dimension = (nsamples, 1, 224,224) # output is simple a mask
# please note that, the fcn model does not assume a fixed row and col size.
Data_train = np.zeros(input_dimension)
Label_train = np.zeros(output_dimension)

model = buildMixModel()
for iteration in range(1, 200):
    for X_batch, Y_batch in batchflow( batch_size,Data_train ,Label_train ):
        print()
        print('-' * 50)
        print('Iteration', iteration)
        #loss = model.fit({'input': X_train}, {'output_mask':Y_train})
        loss = model.train_on_batch({'input': X_batch}, {'output_mask': Y_batch})
        print loss
        # you can save your model here 
        model.save_weights(weightspath,overwrite = 1)

# testing is easy, just load the weights and do the prediction
#model.load_weights(weightspath)
#testinglabel = model.predict({'input': testingbatch})

