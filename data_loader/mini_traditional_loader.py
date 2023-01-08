import matplotlib.pyplot as plt
import os
from tensorflow import keras
import tensorflow as tf
from keras.preprocessing.image import load_img,img_to_array, ImageDataGenerator
import numpy as np
from tensorflow.keras.utils import to_categorical
import tensorflow
import os


class MINI_Traditional_DataGenerator(tensorflow.keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(
        self,data_path='datasets\mini-imagenet',
        num_batch=32,data_type = 'seen',augmentation=False,
        target_size=(64,64)):
        'Initialization'
        self.data_path = data_path
        self.num_batch = num_batch
        self.data_type = data_type
        self.target_size = target_size
        home = os.getcwd()
        dataset_path = os.path.join(home,self.data_path)
        self.seen_path = os.path.join(dataset_path,"seen")
        self.unseen_path = os.path.join(dataset_path,"unseen")
        self.all_path = os.path.join(dataset_path,"all")
        self.augmentation = augmentation
        if self.data_type == "all":
            self.folders = os.listdir(self.all_path)
        elif self.data_type == "seen":
            self.folders = os.listdir(self.seen_path)
        elif self.data_type == "unseen":
            self.folders = os.listdir(self.unseen_path)
        
        self.transformer = ImageDataGenerator(
            width_shift_range=0.1,
            height_shift_range=0.1,
            zoom_range=0.2,
            rotation_range=20,
            horizontal_flip=False,
            shear_range=0.1

        )
        
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(50000 / (self.num_batch))


    def __getitem__(self,index):
        'Generate one batch of data'
        # Generate data
        classes = np.random.choice(self.folders,size=len(self.folders),replace=False)
    
        if self.data_type == "all":
            cls_paths = [os.path.join(self.all_path,C) for C in classes]
        elif self.data_type == "seen":
            cls_paths = [os.path.join(self.seen_path,C) for C in classes]
        elif self.data_type == "unseen":
            cls_paths = [os.path.join(self.unseen_path,C) for C in classes]
        X_sample, label = self.__data_generation(cls_paths)
        #way = np.ones((self.way * self.shot, 1)) * self.way


        return X_sample, label

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        pass

    def __data_generation(self,cls_paths):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        ids = np.random.randint(0,len(cls_paths),self.num_batch)
        X = []
        y = []
        for idx in ids:
            file_path = os.path.join(self.seen_path,cls_paths[idx])
            files = os.listdir(file_path)
            file_name = os.path.join(file_path,files[np.random.randint(len(files))])

            if self.augmentation:
                X += [self.transformer.random_transform(img_to_array(load_img(file_name),target_size=self.target_size))]
            else:
                X += [img_to_array(load_img(os.path.join(file_name),target_size=self.target_size))]
            y += [int(idx)]
            
        X = np.array(X)
        y = np.array(y)
        y = to_categorical(y.flatten(),num_classes=64)

        return X,y