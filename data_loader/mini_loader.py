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
        self,N,Ks,Kq,data_path='datasets\mini-imagenet',
        num_episode=500,data_type = 'all',augmentation=False,
        target_size=(64,64)):
        'Initialization'
        self.N = N
        self.Ks = Ks
        self.Kq = Kq
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
        classes = np.random.choice(self.folders,size=self.N,replace=False)
    
        if self.data_type == "all":
            cls_paths = [os.path.join(self.all_path,C) for C in classes]
        elif self.data_type == "seen":
            cls_paths = [os.path.join(self.seen_path,C) for C in classes]
        elif self.data_type == "unseen":
            cls_paths = [os.path.join(self.unseen_path,C) for C in classes]
        X_sample, X_query, label = self.__data_generation(self.Ks,self.Kq,cls_paths)
        #way = np.ones((self.way * self.shot, 1)) * self.way


        return [X_sample, X_query], label

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        pass

    def __data_generation(self,Ks,Kq,cls_paths):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        
        Dsupport = []
        Dquery = []
        label_query = []
        ql = 0
        K = Ks + Kq
    
        for cls_path in cls_paths:
            samples = os.listdir(cls_path)
            idx = np.random.choice(len(samples),K,replace=False)
            idx_s = np.random.choice(idx,Ks,replace=False)
            idx_q = [x for x in idx if x not in idx_s]
            data_s = np.array(samples)[idx_s]
            data_q = np.array(samples)[idx_q]
            _label = []
            _label_q = []
            X = []
            if self.augmentation:
                X += [self.transformer.random_transform(img_to_array(load_img(os.path.join(cls_path,file),target_size=self.target_size))) for file in data_s]
            else:
                X += [img_to_array(load_img(os.path.join(cls_path,file),target_size=self.target_size)) for file in data_s]
            X = np.array(X)
            Dsupport += [X]
            X = []
            X += [img_to_array(load_img(os.path.join(cls_path,file),target_size=self.target_size)) for file in data_q]
            
            _label_q += [ql for _ in data_q]
            label_query += [_label_q]
            ql = ql + 1
            X = np.array(X)
            Dquery += [X]
        Dsupport = np.array(Dsupport)
        Dquery = np.array(Dquery)
        label_query = np.array(label_query)
        label_query = to_categorical(label_query.flatten(),num_classes=self.N)

        return Dsupport,Dquery,label_query