from tensorflow import keras
from .blocks import res_block
def create_resnet12(input_shape = (64,64,3),weights=None):
    H,W,C = input_shape
    inp = keras.layers.Input([H,W,C])
    x = inp
    x = res_block(x,kernel_size=3,n_filters=64,strides=(1,1))
    x = res_block(x,kernel_size=3,n_filters=128,strides=(1,1))
    x = res_block(x,kernel_size=3,n_filters=256,strides=(1,1))
    x = res_block(x,kernel_size=3,n_filters=512,strides=(1,1))
    #x = keras.layers.Flatten()(x)

    return keras.Model(inputs=[inp],outputs=[x])