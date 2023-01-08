"""
This module generates encoders for feature extracting
"""
from tensorflow import keras
import tensorflow as tf
from tensorflow import keras
from .blocks import conv_block
from .distances import Euclidean_Distance,Cosine_Distance

from tensorflow.keras.applications import  MobileNetV2

def create_mobilenet(input_shape = (64,64,3)):
    inp = keras.layers.Input(input_shape)
    mobilenet = MobileNetV2(input_shape=input_shape,weights='imagenet',include_top=False)
    #mobilenet.summary()
    dens_encoder = keras.Model(inputs=mobilenet.inputs,outputs=mobilenet.get_layer('block_14_add').output)
    x = dens_encoder(inp)
    #x = keras.layers.Conv2D(
        #kernel_size=(1,1),filters=160,padding='same',
        #kernel_regularizer = keras.regularizers.L2(1e-5),
        #bias_regularizer=keras.regularizers.L2(1e-5),
    #kernel_initializer='he_normal')(x)
    #x = keras.layers.BatchNormalization(axis=1)(x)
    #x = keras.layers.Activation('relu')(x)
    #x = conv_block(x,kernel_size=(3,3),n_filters=150,strides=(2,2))
    #x = conv_block(x,kernel_size=(5,5),n_filters=160,strides=(1,1))
    #x = keras.layers.MaxPooling2D(pool_size=(2,2),strides=(2,2))(x)
    x = keras.layers.Flatten()(x)
    return keras.Model(inp,x,name='Encoder')

def create_model(input_shape = (64,64,3)):
    support = keras.layers.Input(input_shape)
    query = keras.layers.Input(input_shape)
    encoder = create_mobilenet(input_shape=input_shape)
    encoder.summary()
    support_features = encoder(support)
    query_features = encoder(query)
    dist = Euclidean_Distance()([support_features,query_features])
    out = keras.layers.Activation("softmax")(dist)
    return keras.Model(inputs = [support,query],outputs=out)
