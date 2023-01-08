"""
This module generates encoders for feature extracting
"""
from tensorflow import keras
import tensorflow as tf
from tensorflow import keras
from .blocks import conv_block
from .distances import Euclidean_Distance

from tensorflow.keras.applications import ResNet50


def create_resnet(input_shape = (64,64,3)):
    H,W,C = input_shape
    inp = keras.layers.Input([H,W,C])
    resnet = ResNet50(input_shape=input_shape,weights='imagenet',include_top=False)
    res_encoder = keras.Model(inputs=resnet.inputs,outputs=resnet.get_layer('conv3_block2_out').output,name='backbone')
    x = res_encoder(inp)

    x = keras.layers.Conv2D(
        kernel_size=(1,1),filters=256,padding='same',
        kernel_regularizer = keras.regularizers.L2(1e-5),
        bias_regularizer=keras.regularizers.L2(1e-5),
    kernel_initializer='he_normal')(x)
    x = conv_block(x,kernel_size=(5,5),n_filters=256,strides=(2,2))
    
    x = keras.layers.Flatten()(x)
    return keras.Model(inp,x,name='Encoder')

def create_model(input_shape = (64,64,3)):
    support = keras.layers.Input(input_shape)
    query = keras.layers.Input(input_shape)
    encoder = create_resnet(input_shape=input_shape)
    support_features = encoder(support)
    query_features = encoder(query)
    dist = Euclidean_Distance()([support_features,query_features])
    out = keras.layers.Activation("softmax")(dist)
    return keras.Model(inputs = [support,query],outputs=out)
