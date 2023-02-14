from tensorflow import keras
from models.densenet import create_densenet
import tensorflow as tf
import numpy as np
from tensorflow.keras.metrics import CategoricalAccuracy
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import load_img,img_to_array
from tensorflow.keras.utils import to_categorical
import pandas as pd
from tensorflow.keras.models import load_model
from models.stn import BilinearInterpolation,Localization
from argparse import ArgumentParser

parser = ArgumentParser('SENet GTSRB Benchmark')
parser.add_argument('--run',type=str,default='test')
parser.add_argument('--epochs',type=int,default=20)

args = parser.parse_args()

def standardize(img):
    mean = np.mean(img)
    std = np.std(img)
    return (img-mean)/std

def make_gtsrb_model():
    input_shape = (64,64,3)
    encoder = create_densenet(input_shape=input_shape)
    inp = keras.layers.Input(input_shape)
    x = encoder(inp)
    x = keras.layers.Dense(43,kernel_initializer='he_normal')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation('softmax')(x)
    return keras.Model(inputs=inp,outputs = x)

def generate_data(subset='train'):
    if subset == 'train':
        datapath = 'datasets/GTSRB/all'
        datagen = ImageDataGenerator(preprocessing_function=standardize)
        return datagen.flow_from_directory(
            datapath,batch_size=512,class_mode='categorical',
            target_size=(64,64))
    if subset == 'test':
        test_path = 'datasets/GTSRB/GTSRB_Test'
        X = []
        print(f'loading GTSRB Test data ...')
        files = os.listdir(test_path)
        files.sort()
        for f in files:
            file_path = os.path.join(test_path,f)
            X += [standardize(
                img_to_array(
                    load_img(
                        file_path,target_size=(64,64),
                        interpolation='bilinear')))]
        X = tf.constant(np.array(X))
        df = pd.read_csv('datasets/GTSRB/GT-final_test.csv',sep=';')
        y_true = tf.constant(
            to_categorical(np.array(df['ClassId']),num_classes=43))
        return X,y_true

def train(epochs):
    senet_gtsrb = make_gtsrb_model()
    metric = CategoricalAccuracy()
    optimizer = keras.optimizers.Adam(learning_rate=0.001,epsilon=1e-8)
    loss_fun = keras.losses.CategoricalCrossentropy()
    senet_gtsrb.compile(optimizer=optimizer,loss=loss_fun,metrics=metric)
    train_gen = generate_data(subset ='train')
    X,y_true = generate_data(subset = 'test')

    best_test_acc = 0
    for e in range(epochs):
        print(f'========epoch {e+1}=========')
        senet_gtsrb.fit(train_gen,epochs=1)
        _,test_acc = senet_gtsrb.evaluate(X,y_true,batch_size=128,verbose=0)
        print(f'{test_acc:.4f}')
        if test_acc > best_test_acc:
            senet_gtsrb.save('model_files/senet_gtsrb_bench.h5')
            best_test_acc = test_acc
            
def test():
    final_h5 = 'model_files/best_models/senet_gtsrb_bench.h5'
    senet_gtsrb = load_model(
        final_h5,
        custom_objects={
        'BilinearInterpolation':BilinearInterpolation,
        'Localization':Localization},compile=True)
    metric = CategoricalAccuracy()
    senet_gtsrb.compile(metrics=metric)
    X,y_true = generate_data(subset = 'test')
    senet_gtsrb.evaluate(X,y_true,batch_size=256,verbose=1)

if __name__ == '__main__':
    if args.run == 'train':
        train(args.epochs)
    if args.run == ('test'):
        test()