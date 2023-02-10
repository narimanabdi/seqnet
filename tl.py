from tensorflow import keras
import tensorflow as tf

import numpy as np

from tensorflow.keras.metrics import CategoricalAccuracy

import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from utils import load_config
from gpu.gpu import set_gpu_memory_growth
import argparse
from tensorflow.keras.preprocessing.image import load_img,img_to_array, ImageDataGenerator
import numpy as np
from tensorflow.keras.utils import to_categorical
import pandas as pd
from sklearn.metrics import accuracy_score
from data_loader import get_loader
from tensorflow.keras.models import load_model
from models.stn import BilinearInterpolation,Localization
from models.distances import Weighted_Euclidean_Distance
from tensorflow import image

def standardize(img):
    mean = np.mean(img)
    std = np.std(img)
    return (img - mean) / std





datapath_tr = 'datasets/TT100K/template/all'
datapath_te = 'datasets/TT100K/all'
#datagen = ImageDataGenerator(preprocessing_function=standardize,validation_split=0.1)
#train_gen = datagen.flow_from_directory(datapath,batch_size=256,class_mode='categorical',target_size=(64,64),subset='training')
#val_gen = datagen.flow_from_directory(datapath,batch_size=256,class_mode='categorical',target_size=(64,64),subset='validation')

fdatagen = ImageDataGenerator(preprocessing_function=standardize)
tdatagen = ImageDataGenerator(preprocessing_function=standardize,rotation_range=45.0)
tr_gen = tdatagen.flow_from_directory(datapath_tr,batch_size=36,class_mode='categorical',target_size=(64,64))
te_gen = fdatagen.flow_from_directory(datapath_te,batch_size=128,class_mode='categorical',target_size=(64,64))

def test(model,data):
    x,y = data
    indexes = np.arange(len(x))
    batch = 256
    iterations = len(x) // batch
    acc = 0
    pb = tf.keras.utils.Progbar(iterations,verbose=1)
    for i in range(iterations):
        y_pred = model(x[indexes[i*batch:(i+1)*batch]])
        acc += np.mean(keras.metrics.categorical_accuracy(y[i*batch:(i+1)*batch],y_pred))
        print(acc)
        pb.add(1)
    return acc / iterations


if __name__ == "__main__":
    encoder_h5 = 'model_files/best_encoders/w_densenet_gtsrb2tt100k_encoder.h5'
    loaded_encoder = load_model(
        encoder_h5,
        custom_objects={
            'BilinearInterpolation':BilinearInterpolation,
            'Localization':Localization},compile=False)
    loaded_encoder.trainable = False

    inp = keras.layers.Input((64,64,3))
    x = loaded_encoder(inp)
    #x = x = keras.layers.Dense(100,activation = 'relu',kernel_initializer='he_normal')(x)
    x = keras.layers.Dense(36,activation = 'softmax',kernel_initializer='he_normal')(x)
    clf_encoder = keras.Model(inputs=inp,outputs = x)

    metric = CategoricalAccuracy()
    train_loss_tracker = keras.metrics.Mean(name='loss')
    #optimizer = keras.optimizers.SGD(learning_rate=0.1,momentum=0.9)
    optimizer = keras.optimizers.Adam(learning_rate=0.001)
    loss_fun = keras.losses.CategoricalCrossentropy()
    clf_encoder.compile(optimizer=optimizer,loss=loss_fun,metrics=metric)

    for e in range(100):
        clf_encoder.fit(tr_gen,epochs=1)
        _,acc = clf_encoder.evaluate(te_gen)
        #acc = test(clf_encoder,[X,y_true])
        print(f'test accuracy = {acc:.4f}')
        
    
        