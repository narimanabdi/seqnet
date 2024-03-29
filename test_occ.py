from tensorflow.keras.metrics import CategoricalAccuracy
import tensorflow as tf
from data_loader import get_loader
from tensorflow import keras
import matplotlib.pyplot as plt
from tensorflow.keras.utils import to_categorical
import numpy as np
from models.distances import Euclidean_Distance
from models.stn import BilinearInterpolation,Localization
import os
#from models.metrics import accuracy
from tensorflow.keras.metrics import Mean
from sklearn.metrics.pairwise import euclidean_distances as eudist
from sklearn.metrics.pairwise import cosine_distances as codist
from models.senet import Senet
from prettytable import PrettyTable
from tensorflow.keras.preprocessing.image import load_img,img_to_array
import pandas as pd
from tensorflow.keras.metrics import Mean
from PIL import Image, ImageDraw
import numpy as np
from keras.metrics import Recall, Precision,TruePositives

tp = TruePositives()

acc_tracker = Mean(name='Nearest_Neighbor_Accuracy')
rec_tracker = Mean(name='Nearest_Neighbor_Recall')
pre_tracker = Mean(name='Nearest_Neighbor_Precision')
_recall = Recall()
_precision = Precision()

dist_fn = Euclidean_Distance()

def standardize(img):
    mean = np.mean(img)
    std = np.std(img)
    return (img - mean) / std

def generate_occluded_GTSRB():

    df = pd.read_csv('datasets/GTSRB/GT-final_test.csv',sep=';')

    test_path = 'datasets/GTSRB/GTSRB_occ_Test'
    print(f'loading occluded GTSRB test labels ...')
    files = os.listdir(test_path)
    files.sort()
    file_names = []
    for i,f in enumerate(files):
        file_names += [f]
    
    test_df = pd.DataFrame()
    for name in file_names:
        temp = pd.DataFrame(df.loc[df['Filename'] == name])
        test_df = pd.concat([test_df,temp],axis = 0)

    X = []
    print(f'loading occluded GTSRB test data ...')
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
    y_true = tf.constant(to_categorical(np.array(test_df['ClassId']),num_classes=43))
    return X,y_true


@tf.function
def nn(model,inp,ztemplates):
    z = model(tf.expand_dims(inp,axis=0))
    return dist_fn([ztemplates,z])

def test(encoder_h5,data):
    X,y_true = data
    loaded_encoder = keras.models.load_model(
        encoder_h5,
        custom_objects={
            'BilinearInterpolation':BilinearInterpolation,
            'Localization':Localization},compile=False)
    loader = get_loader('gtsrb2tt100k') 
    print(loader)
    test_generator,_ = loader.get_generator(batch=4,dim=64)
    t = iter(test_generator)
    [Xs,_Xq],_y = next(t)
    del _y,_Xq,t
    Zs = loaded_encoder(Xs)
    del Xs
    print('\033[0;32mStart Nearest Neighbor Evaluation\033[0m')
    #pb = tf.keras.utils.Progbar(len(test_generator),verbose=1)
    preds = []
    targets = []
    predictions = np.zeros(43)
    true_positive = np.zeros(43)
    total = np.zeros(43)
    ITER = len(X)
    z = 0
    for i,x in enumerate(X):
        p = nn(loaded_encoder,x,Zs)
        total[np.argmax(y_true[i])] += 1
        predictions[np.argmax(p)] += 1
        if np.argmax(p) == np.argmax(y_true[i]):
            acc_tracker.update_state(1.0)
            true_positive[np.argmax(p)] += 1
        else:
            acc_tracker.update_state(0.0)

        preds += [keras.utils.to_categorical(np.argmax(p), num_classes = 43)]

        targets += [keras.utils.to_categorical(np.argmax(y_true[i]), num_classes = 43)]

        pre_tracker.update_state(_precision(preds,targets))  
        rec_tracker.update_state(_recall(preds,targets))  
        print(f'image {i+1}/{ITER}, accuracy:{acc_tracker.result()*100.0:.2f}, precision:{pre_tracker.result()*100.0:.2f}, recall:{rec_tracker.result()*100.0:.2f}')
    print('hello')
    print(true_positive.sum()/predictions.sum())

if __name__ == '__main__':
    encoder_file = 'model_files/best_encoders/densenet_gtsrb2tt100k_encoder.h5'
    x,y = generate_occluded_GTSRB()
    test(encoder_h5=encoder_file,data = (x,y))
    print('heloo')
    