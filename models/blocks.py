from tensorflow import keras
import tensorflow as tf
from tensorflow import keras

def conv_block(input_tensor,kernel_size,n_filters,strides,padding='valid'):
    x = keras.layers.Conv2D(
        kernel_size=kernel_size,
        filters=n_filters,
        padding=padding,strides=strides,
        kernel_regularizer = keras.regularizers.L2(1e-5),
        bias_regularizer=keras.regularizers.L2(1e-5),
        kernel_initializer='he_normal')(input_tensor)
    x = keras.layers.BatchNormalization(axis=1)(x)
    #x = keras.layers.LeakyReLU()(x)
    x = keras.layers.Activation('relu')(x)
    #x = keras.layers.Dropout(0.1)(x)
    return x

def conv_block_zp(input_tensor,kernel_size,n_filters,strides,padding=(1,1)):
    x = keras.layers.ZeroPadding2D(padding)(input_tensor)
    x = keras.layers.Conv2D(
        kernel_size=kernel_size,
        filters=n_filters,
        padding='valid',strides=strides,
        kernel_regularizer = keras.regularizers.L2(1e-4),
        bias_regularizer=keras.regularizers.L2(1e-4),
        kernel_initializer='he_normal')(x)
    x = keras.layers.BatchNormalization(axis=1)(x)
    x = keras.layers.LeakyReLU()(x)
    #x = keras.layers.Activation('relu')(x)
    #x = keras.layers.Dropout(0.2)(x)
    return x

def conv_block_di(input_tensor,kernel_size,n_filters,dilation_rate):
    x = keras.layers.Conv2D(
        kernel_size=kernel_size,
        filters=n_filters,
        padding='same',dilation_rate=dilation_rate,
        kernel_regularizer = keras.regularizers.L2(5e-4),
        bias_regularizer=keras.regularizers.L2(5e-4))(input_tensor)
    x = keras.layers.BatchNormalization(axis=1)(x)
    x = keras.layers.Activation('relu')(x)
    x = keras.layers.Dropout(0.1)(x)

    x = keras.layers.Conv2D(
        kernel_size=(1,1),
        filters=n_filters,
        padding='same',
        kernel_regularizer = keras.regularizers.L2(5e-4),
        bias_regularizer=keras.regularizers.L2(5e-4))(input_tensor)
    x = keras.layers.BatchNormalization(axis=1)(x)

    return keras.layers.Activation('relu')(x)

def atn(input_tensor,kernel_size,n_filters):
    n_f1, n_f2 = n_filters
    x1 = keras.layers.Conv2D(
        kernel_size=kernel_size,filters=n_f1,padding='same',
        dilation_rate=(1,1),
        kernel_regularizer = keras.regularizers.L2(5e-4),
        bias_regularizer=keras.regularizers.L2(5e-4))(input_tensor)
    
    
    x2 = keras.layers.Conv2D(
        kernel_size=kernel_size,filters=n_f1,padding='same',
        dilation_rate=(2,2),
        kernel_regularizer = keras.regularizers.L2(5e-4),
        bias_regularizer=keras.regularizers.L2(5e-4))(input_tensor)
    
    x3 = keras.layers.Conv2D(
        kernel_size=kernel_size,filters=n_f1,padding='same',
        dilation_rate=(3,3),
        kernel_regularizer = keras.regularizers.L2(5e-4),
        bias_regularizer=keras.regularizers.L2(5e-4))(input_tensor)

    x4 = keras.layers.Conv2D(
        kernel_size=kernel_size,filters=n_f1,padding='same',
        dilation_rate=(4,4),
        kernel_regularizer = keras.regularizers.L2(5e-4),
        bias_regularizer=keras.regularizers.L2(5e-4))(input_tensor)

    
    x = keras.layers.Concatenate()([x1,x2,x3,x4])
    x = keras.layers.BatchNormalization(axis=1)(x)
    #x = keras.layers.LeakyReLU()(x)
    return keras.layers.Activation('relu')(x)

def sam(input_tensor):
    f = keras.layers.MaxPooling2D(pool_size=(2,2),strides=(2,2))(input_tensor)
    f = conv_block(f,kernel_size=(1,1),n_filters=256,strides=(1,1))

    g = keras.layers.AveragePooling2D(pool_size=(2,2),strides=(2,2))(input_tensor)
    g = conv_block(g,kernel_size=(1,1),n_filters=256,strides=(1,1))

    h = keras.layers.MaxPooling2D(pool_size=(2,2),strides=(2,2))(input_tensor)    
    h = conv_block(h,kernel_size=(1,1),n_filters=256,strides=(1,1))
    
    k = conv_block(input_tensor,kernel_size=(2,2),n_filters=256,strides=(2,2))

    fog = keras.layers.Multiply()([f,g])
    fog = keras.layers.Activation('softmax')(fog)

    fogoh = keras.layers.Multiply()([fog,h])

    x = keras.layers.Add()([k,fogoh])
    x = keras.layers.BatchNormalization(axis=1)(x)

    return keras.layers.Activation('relu')(x)

def res_block(input_tensor, kernel_size, n_filters,strides=(1,1)):
    x = keras.layers.Conv2D(n_filters, kernel_size, padding='same',
                      kernel_initializer='he_normal',strides=strides)(input_tensor)
    x = keras.layers.BatchNormalization(axis=1)(x)
    x = keras.layers.LeakyReLU()(x)
    x = keras.layers.Conv2D(n_filters, kernel_size, padding='same',
                      kernel_initializer='he_normal',strides=strides)(x)
    x = keras.layers.BatchNormalization(axis=1)(x)
    x = keras.layers.LeakyReLU()(x)
    x = keras.layers.Conv2D(n_filters, kernel_size, padding='same',
                      kernel_initializer='he_normal',strides=strides)(x)
    x = keras.layers.BatchNormalization(axis=1)(x)

    shortcut = keras.layers.Conv2D(n_filters, kernel_size=(1,1), padding='same',
                      kernel_initializer='he_normal',strides=strides)(input_tensor)
    shortcut = keras.layers.BatchNormalization(axis=1)(shortcut)


    x = keras.layers.add([x,shortcut])
    x = keras.layers.LeakyReLU()(x)
    x = keras.layers.MaxPooling2D((2, 2), strides=(2, 2))(x)
    return x

def res18_block(input_tensor, kernel_size, n_filters,strides=(1,1)):
    x = keras.layers.Conv2D(n_filters, kernel_size, padding='same',
                      kernel_initializer='he_normal',strides=strides)(input_tensor)
    x = keras.layers.BatchNormalization(axis=1)(x)
    x = keras.layers.LeakyReLU()(x)
    x = keras.layers.Conv2D(n_filters, kernel_size, padding='same',
                      kernel_initializer='he_normal',strides=(1,1))(x)
    x = keras.layers.BatchNormalization(axis=1)(x)

    shortcut = keras.layers.Conv2D(n_filters, kernel_size=(1,1), padding='same',
                      kernel_initializer='he_normal',strides=strides)(input_tensor)
    shortcut = keras.layers.BatchNormalization(axis=1)(shortcut)


    x = keras.layers.add([x,shortcut])
    x = keras.layers.LeakyReLU()(x)
    #x = keras.layers.MaxPooling2D((2, 2), strides=(2, 2))(x)
    return x