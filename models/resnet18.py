from tensorflow import keras
from .blocks import res18_block
def create_resnet18(input_shape = (64,64,3),weights=None):
    H,W,C = input_shape
    inp = keras.layers.Input([H,W,C])
    x = keras.layers.ZeroPadding2D(padding=(3, 3), name='conv1_pad')(inp)
    x = keras.layers.Conv2D(64, (7, 7),
                      strides=(2, 2),
                      padding='valid',
                      kernel_initializer='he_normal',
                      name='conv1')(x)
    x = keras.layers.BatchNormalization(axis=1, name='bn_conv1')(x)
    x = keras.layers.LeakyReLU()(x)
    x = keras.layers.ZeroPadding2D(padding=(1, 1), name='pool1_pad')(x)
    x = keras.layers.MaxPooling2D((3, 3), strides=(2, 2))(x)
    x = res18_block(x,kernel_size=3,n_filters=64,strides=(1,1))
    x = res18_block(x,kernel_size=3,n_filters=64,strides=(1,1))


    x = res18_block(x,kernel_size=3,n_filters=128,strides=(2,2))
    x = res18_block(x,kernel_size=3,n_filters=128,strides=(1,1))


    x = res18_block(x,kernel_size=3,n_filters=256,strides=(2,2))
    x = res18_block(x,kernel_size=3,n_filters=256,strides=(1,1))


    x = res18_block(x,kernel_size=3,n_filters=512,strides=(2,2))
    x = res18_block(x,kernel_size=3,n_filters=512,strides=(1,1))

    #x = keras.layers.Flatten()(x)

    return keras.Model(inputs=[inp],outputs=[x])