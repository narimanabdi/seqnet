import matplotlib.pyplot as plt
import os
from tensorflow import keras
import tensorflow as tf
from keras.preprocessing.image import load_img,img_to_array, ImageDataGenerator
import numpy as np
from tensorflow.keras.utils import to_categorical
import tensorflow
import os


class MINI_DataGenerator(tensorflow.keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(
        self,n_way,k_shot,n_query,data_path='datasets\mini-imagenet',
        num_episode=500,data_type = 'all',augmentation=False,
        target_size=(64,64)):
        'Initialization'
        self.n_way = n_way
        self.k_shot = k_shot
        self.n_query = n_query
        self.data_path = data_path
        self.num_episode = num_episode
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
        return self.num_episode


    def __getitem__(self,index):
        'Generate one batch of data'
        # Generate data
        classes = np.random.choice(self.folders,size=self.n_way,replace=False)
    
        if self.data_type == "all":
            cls_paths = [os.path.join(self.all_path,C) for C in classes]
        elif self.data_type == "seen":
            cls_paths = [os.path.join(self.seen_path,C) for C in classes]
        elif self.data_type == "unseen":
            cls_paths = [os.path.join(self.unseen_path,C) for C in classes]
        X_sample, X_query, label = self.__data_generation(cls_paths)
        #way = np.ones((self.way * self.shot, 1)) * self.way


        return [X_sample, X_query], label

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        pass

    def __data_generation(self,cls_paths):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        
        X = np.empty((self.n_way * self.k_shot + self.n_query,*self.target_size,3))
        #labels = np.empty((self.n_way * self.k_shot + self.n_query,1))
        labels = np.empty((self.n_query,1))
        for cls_path in cls_paths:
            samples = os.listdir(cls_path)
            idx = np.random.choice(len(samples),self.k_shot,replace=False)
            data = np.array(samples)[idx]
            i = 0
            for d in data:
                if self.augmentation:
                    X[i] = self.transformer.random_transform(img_to_array(load_img(os.path.join(cls_path,d),target_size=self.target_size)))
                else:
                    X[i] = img_to_array(load_img(os.path.join(cls_path,d),target_size=self.target_size))
                i += 1
            
        i = self.n_way * self.k_shot
        lidx = 0
        for _ in range(self.n_query):
            cls_idx = np.random.randint(self.n_way)
            c = cls_paths[cls_idx]
            samples = os.listdir(c)
            data = samples[np.random.randint(len(samples))]
            qfile = os.path.join(c,str(data))
            X[i] = img_to_array(load_img(qfile,target_size=(64,64)))
            labels[lidx] = cls_idx
            i += 1
            lidx += 1
        labels = to_categorical(labels,num_classes=self.n_way)

        return X[:self.n_way * self.k_shot],X[self.n_way * self.k_shot:],labels