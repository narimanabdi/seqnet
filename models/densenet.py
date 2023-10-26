# This module generates encoders for feature extracting
from tensorflow import keras
from tensorflow.keras.layers import Input,Flatten,Dense,MaxPooling2D,Conv2D
from tensorflow.keras.layers import BatchNormalization, Activation
from .blocks import conv_block
from .distances import Weighted_Euclidean_Distance, Mahalanobis_Distance
from .stn import stn
from tensorflow.keras.applications import DenseNet121
from models.senet import Senet
import tensorflow as tf

def count_layers(model):
   num_layers = len(model.layers)
   for layer in model.layers:
      if isinstance(layer, tf.keras.Model):
         num_layers += count_layers(layer)
   return num_layers

def create_densenet(input_shape = (64,64,3),dense_truncated_layer='conv3_block2_concat'):
    inp = Input(input_shape)
    
    densnet = DenseNet121(
        input_shape=input_shape,weights='imagenet',include_top=False)
    #densnet.summary()
    #print(f'layer of base is {count_layers(densnet)}')
    dens_encoder = keras.Model(
        inputs=densnet.inputs,
        outputs=densnet.get_layer(dense_truncated_layer).output)#conv3_block2_concat
    #print(f'layer of GFENet is {count_layers(dens_encoder)}')
    x = stn(inp,[64,64,100],kernel_size=(5,5),stage=1)
    x = dens_encoder(x)
    x = Conv2D(
        kernel_size=(1,1),filters=100,padding='same',
        kernel_initializer='he_normal')(x)
    x = stn(x,[100,100,100], stage=2)
    x = conv_block(x,kernel_size=(3,3),n_filters=100,strides=(1,1))
    x = MaxPooling2D(pool_size=(2,2),strides=(2,2))(x)
    x = Flatten()(x)
    x = Dense(units = 300,kernel_initializer="he_normal")(x)
    x = BatchNormalization(axis=-1)(x)
    x = Activation('linear')(x)
    return keras.Model(inp,x,name='encoder')

def create_model(input_shape = (64,64,3),truncated_layer='conv3_block2_concat'):
    support = Input(input_shape)
    query = Input(input_shape)
    encoder = create_densenet(input_shape=input_shape,dense_truncated_layer=truncated_layer)
    #encoder.summary()
    support_features = encoder(support)
    query_features = encoder(query)
    dist = Mahalanobis_Distance()([support_features,query_features])
    out = Activation("softmax")(dist)
    return Senet(inputs = [support,query],outputs=out)
