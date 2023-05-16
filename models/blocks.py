# This file contains some model architecture blocks
from tensorflow.keras.layers import Conv2D,BatchNormalization,Activation

#Convolutional block with BachNormalization and ReLU activation function
def conv_block(input_tensor,kernel_size,n_filters,strides,padding='valid'):
    x = Conv2D(
        kernel_size=kernel_size,
        filters=n_filters,
        padding=padding,strides=strides,
        kernel_initializer='he_normal')(input_tensor)
    x = BatchNormalization(axis=-1)(x)
    x = Activation('relu')(x)
    return x