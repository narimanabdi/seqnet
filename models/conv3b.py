"""
This module generates encoders for feature extracting
"""
from tensorflow import keras
import tensorflow as tf
from tensorflow import keras

from tensorflow.keras.applications import DenseNet121

class Distance_Layer(keras.layers.Layer):
    def __init__(self,**kwargs):
        super(Distance_Layer,self).__init__(**kwargs)
        #self.constant = tf.Variable(initial_value=0.1, dtype=tf.float32, trainable=True)

    def call(self,inputs,**kwargs):
        support,query = inputs
        #print(query.shape)
        #print(v.shape)
        q2 = tf.reduce_sum(query ** 2, axis=1, keepdims=True)
        s2 = tf.reduce_sum(support ** 2, axis=1, keepdims=True)
        qdots = tf.matmul(query, tf.transpose(support))
        return  -(tf.sqrt(q2 + tf.transpose(s2) - 2 * qdots))
        #return  -(self.constant * (q2 + tf.transpose(s2) - 2 * qdots))
        #return  -((q2 + tf.transpose(s2) - 2 * qdots))
    
    def compute_output_shape(self,input_shape):
        return tf.TensorShape(input_shape[:-1])
    
    def get_config(self):
        config = super(Distance_Layer,self).get_config()
        return config

def conv_block(input_tensor,kernel_size,n_filters,strides,padding='valid'):
    x = keras.layers.Conv2D(
        kernel_size=kernel_size,
        filters=n_filters,
        padding=padding,strides=strides,
        kernel_regularizer = keras.regularizers.L2(1e-4),
        bias_regularizer=keras.regularizers.L2(1e-4),
        kernel_initializer='he_normal')(input_tensor)
    x = keras.layers.BatchNormalization(axis=1)(x)
    #x = keras.layers.LeakyReLU()(x)
    x = keras.layers.Activation('relu')(x)
    #x = keras.layers.Dropout(0.1)(x)
    return x

def create_conv3b(input_shape = (32,32,3)):
    inp = keras.layers.Input(input_shape)
    x = conv_block(inp,kernel_size=(5,5),n_filters=100,strides=(1,1))
    x = keras.layers.MaxPooling2D(pool_size=(3,3),strides=2)(x)

    x = conv_block(x,kernel_size=(5,5),n_filters=100,strides=(1,1))
    x = keras.layers.MaxPooling2D(pool_size=(3,3),strides=2)(x)

    x = conv_block(x,kernel_size=(4,4),n_filters=100,strides=(1,1))
    x = keras.layers.MaxPooling2D(pool_size=(3,3),strides=2)(x)
    
    x = keras.layers.Flatten()(x)
    return keras.Model(inp,x)

def create_model(input_shape = (64,64,3)):
    support = keras.layers.Input(input_shape)
    query = keras.layers.Input(input_shape)
    encoder = create_conv3b(input_shape=input_shape)
    support_features = encoder(support)
    query_features = encoder(query)
    dist = Distance_Layer()([support_features,query_features])
    out = keras.layers.Activation("softmax")(dist)
    return keras.Model(inputs = [support,query],outputs=out)
