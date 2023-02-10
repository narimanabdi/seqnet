#####################
#                   #
#     Meta Test     #
#                   #
#####################
from time import time
from data_loader import get_loader
from tensorflow.keras.models import load_model
from models.stn import BilinearInterpolation,Localization
from sklearn.neighbors import KNeighborsClassifier
from tensorflow.keras.utils import to_categorical
import numpy as np
import tensorflow as tf
import pandas as pd
from tensorflow.keras.preprocessing.image import load_img,img_to_array
from tensorflow.keras.utils import to_categorical
import os

def standardize(img):
    mean = np.mean(img)
    std = np.std(img)
    return (img - mean) / std


if __name__ == '__main__':
    home = os.getcwd()
    temp_path = os.path.join(home,'datasets/GTSRB/template/all')
    classes = np.asarray(os.listdir(temp_path))
    Xs = np.empty((43,64,64,3))
    for i,C in enumerate(classes):
        c_path = os.path.join(temp_path,C)
        for f in os.listdir(c_path):
            file_path = os.path.join(c_path,f)
            Xs[i] = standardize(
                img_to_array(
                    load_img(file_path,target_size=(64,64,3),
                    interpolation='bilinear')))
    test_path = 'datasets/GTSRB/GTSRB_Test'
    X = []
    print(f'loading GTSRB Test data ...')
    for f in os.listdir(test_path):
        file_path = os.path.join(test_path,f)
        X += [standardize(
            img_to_array(
                load_img(
                    file_path,target_size=(64,64),
                    interpolation='bilinear')))]
    X = np.array(X)
    df = pd.read_csv('datasets/GTSRB/GT-final_test.csv',sep=';')
    y_true = to_categorical(np.array(df['ClassId']),num_classes=43)

    #encoder_h5 = 'model_files/best_encoders/densenet_' + args.test + '_encoder.h5'
    encoder_h5 = 'model_files/best_encoders/densenet_gtsrb2tt100k_encoder.h5'
    loaded_encoder = load_model(
        encoder_h5,
        custom_objects={
            'BilinearInterpolation':BilinearInterpolation,
            'Localization':Localization},compile=False)

    Zs = loaded_encoder(Xs)
    knn = KNeighborsClassifier(n_neighbors=1,n_jobs=-1,metric='l2')
    y_train = to_categorical(np.arange(43),num_classes=43)
    knn.fit(Zs,y_train);
    start = time()
    total_samples = len(X)
    acc = 0
    pb = tf.keras.utils.Progbar(total_samples,verbose=1)
    for i,test_image in enumerate(X):
        z_test = loaded_encoder(tf.expand_dims(test_image,axis=0))
        y_test = tf.expand_dims(y_true[i],axis=0)
        #y_pred = knn.predict(z_test)
        acc += knn.score(z_test,y_test)
        #if np.argmax(y_pred) == np.argmax(y_true[i]):
            #tp += 1
        #acc = tp / total_samples
        pb.add(1)
    end = time() - start
    acc = acc / total_samples

    time_per_image = end / (len(y_true))
    fps = int(1/time_per_image)
    mean_accuracy = np.round(acc * 100,2)

    print('\033[0;31m')
    print('+---------------------------+')
    print('|    1-NN Testing Report    |')
    print('+---------------------------+')
    print(f'|      Test    | GTSRB|')
    print('+---------------------------+')
    print(f'|Mean accuracy |{mean_accuracy:.4f}     |')
    print('+---------------------------+')
    print('+---------------------------+')
    print(f'|FPS |{fps}     |')
    print('+---------------------------+')
    print('+---------------------------+')
    print(f'|Inference time per image |{time_per_image:.4f}     |')
    print('+---------------------------+')
    print('\033[0m')
