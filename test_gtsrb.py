from tensorflow.keras.metrics import CategoricalAccuracy
import tensorflow as tf
from data_loader import get_loader
from tensorflow import keras
import matplotlib.pyplot as plt
from tensorflow.keras.utils import to_categorical
import numpy as np
from models.distances import Euclidean_Distance, Weighted_Euclidean_Distance
from models.stn import BilinearInterpolation,Localization
from sklearn.neighbors import KNeighborsClassifier
from time import time
import os
from prettytable import PrettyTable
#from models.metrics import accuracy
from tensorflow.keras.metrics import Mean
from sklearn.metrics.pairwise import euclidean_distances as eudist
from sklearn.metrics.pairwise import cosine_distances as codist
from models.senet import Senet

from models.utils import make_senet_model, loss_mse
from tensorflow.keras.preprocessing.image import load_img,img_to_array
import pandas as pd
from tensorflow.keras.metrics import Mean

from PIL import Image, ImageDraw
import numpy as np

def add_random_occlusions(image_path, num_occlusions=1, occlusion_size=(15, 15)):
    # Open the original image
    original_image = Image.open(image_path)

    # Create a drawing object
    draw = ImageDraw.Draw(original_image)

    # Get the dimensions of the image
    width, height = original_image.size

    # Add random occlusions
    for _ in range(num_occlusions):
        # Randomly generate occlusion position
        x_pos = np.random.randint(0, width - occlusion_size[0])
        y_pos = np.random.randint(0, height - occlusion_size[1])

        # Randomly generate occlusion color (black in this example)
        occlusion_color = (0, 0, 0)

        # Add the occlusion
        draw.rectangle([x_pos, y_pos, x_pos + occlusion_size[0], y_pos + occlusion_size[1]], fill=occlusion_color)

    # Save the new image
    return original_image

def standardize(img):
    mean = np.mean(img)
    std = np.std(img)
    return (img - mean) / std

@tf.function
def nn(model,inp,ztemplates):
    z = model(tf.expand_dims(inp,axis=0))
    return dist_fn([ztemplates,z])

def run_test(encoder_h5,data):
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
    tval = []
    fpsval = []
    ITER = len(X)
    for i,x in enumerate(X):
        p = nn(loaded_encoder,x,Zs)
        if np.argmax(p) == np.argmax(y_true[i]):
            acc_tracker.update_state(1.0)
        else:
            acc_tracker.update_state(0.0)
        if (i+1) % 1000 == 0:
            print(f'iteration {i+1}/{ITER}, accuracy:{acc_tracker.result()*100.0:.2f}')
    #fps = 1.0 / tmean
    myTable = PrettyTable([" 1-NN Evaluation Report", ""])
    myTable.add_row(["Evaluation Data", 'GTSRB->TT100K'])
    myTable.add_row(["Top-1 Accuracy", f'{acc_tracker.result()*100.0:.2f}'])
    print('\033[0;31m')
    print(myTable)
    print('\033[0m')

def generate_data(subset='train'):
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
    return (X,y_true)

def generate_occluded_data(num_occlusions):

    test_path = 'datasets/GTSRB/GTSRB_Test'
    X = []
    print(f'loading GTSRB Test data ...')
    files = os.listdir(test_path)
    files.sort()
    for f in files:
        input_image_path = os.path.join(test_path,f)
        img = add_random_occlusions(input_image_path, num_occlusions=num_occlusions)
        X += [standardize(tf.image.resize(img_to_array(img),size=(64,64)))]
    X = tf.constant(np.array(X))
    df = pd.read_csv('datasets/GTSRB/GT-final_test.csv',sep=';')
    y_true = tf.constant(
        to_categorical(np.array(df['ClassId']),num_classes=43))
    return (X,y_true)
    
if __name__ == '__main__':
    acc_tracker = Mean(name='Nearest_Neighbor_Accuracy')
    encoder_file = 'model_files/best_encoders/densenet_gtsrb2tt100k_encoder.h5'
    dist_fn = Euclidean_Distance()
    print('========== STARTING Normal Test ==========')
    input_data = generate_data()
    run_test(encoder_h5=encoder_file, data=input_data)
    print('========== STARTING Occlusion Test, num_occlusions:1  ==========')
    input_data = generate_occluded_data(num_occlusions = 1)
    run_test(encoder_h5=encoder_file, data=input_data)
    print('========== STARTING Occlusion Test, num_occlusions:3  ==========')
    input_data = generate_occluded_data(num_occlusions = 3)
    run_test(encoder_h5=encoder_file, data=input_data)
    print('========== STARTING Occlusion Test, num_occlusions:5  ==========')
    input_data = generate_occluded_data(num_occlusions = 5)
    run_test(encoder_h5=encoder_file, data=input_data)