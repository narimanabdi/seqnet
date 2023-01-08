import matplotlib.pyplot as plt
import os
from tensorflow import keras
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img,img_to_array, ImageDataGenerator
import numpy as np
from tensorflow.keras.utils import to_categorical
import tensorflow
import os


class GTSRB_Template_DataGenerator(tensorflow.keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(
        self,n_way,k_shot,n_query,data_path='datasets/GTSRB',
        batch=64,data_type = 'all',augmentation=False,
        target_size=(64,64)):
        'Initialization'
        self.n_way = n_way
        self.k_shot = k_shot
        self.data_path = data_path
        self.batch = batch
        #self.n_query = self.batch - self.n_way * self.k_shot
        self.n_query = n_query
        self.data_type = data_type
        self.target_size = target_size
        self.augmentation = augmentation
        home = os.getcwd()
        #self.parent = os.path.join(self.current_path,os.pardir)
        #self.home = os.path.abspath(os.path.join(self.parent,os.pardir))
        original_path = os.path.join(home,self.data_path)
        #mini_imagenet_path = os.path.join(dataset_path,"GTSRB")
        #original_path = self.data_path
        template_path = os.path.join(original_path,"template")
        
        self.transformer = ImageDataGenerator(
            width_shift_range=0.1,
            height_shift_range=0.1,
            zoom_range=0.2,
            rotation_range=30,
            horizontal_flip=False,
            shear_range=0.1

        )

        if self.data_type == 'seen':
            self.support_path = os.path.join(template_path,"seen")
            self.query_path = os.path.join(original_path,"seen")
        elif self.data_type == 'all':
            self.support_path = os.path.join(template_path,"all")
            self.query_path = os.path.join(original_path,"all")
        else:
            self.support_path = os.path.join(template_path,"unseen")
            self.query_path = os.path.join(original_path,"unseen")

        if self.data_type == 'seen':
            self.classes = np.random.choice(
                os.listdir(self.support_path),
                size=self.n_way,replace=False)
        elif self.data_type == 'all':
            self.classes = np.random.choice(
                os.listdir(self.support_path),
                size=self.n_way,replace=False)
        else:
            self.classes = np.random.choice(
                os.listdir(self.support_path),
                size=self.n_way,replace=False)
        
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        #return self.batch
        samples = 0
        for path in os.listdir(self.query_path):
            file_path = os.path.join(self.query_path,path)
            samples += len(os.listdir(file_path))
        return int(samples / self.batch)


    def __getitem__(self,index):
        'Generate one batch of data'
        # Generate data
        X_sample, X_query, label = self.__data_generation()
        #way = np.ones((self.way * self.shot, 1)) * self.way


        return [X_sample, X_query], label

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        pass

    def __data_generation(self):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        
        X = np.empty((self.n_way * self.k_shot + self.n_query,*self.target_size,3))
        #labels = np.empty((self.n_way * self.k_shot + self.n_query,1))
        labels = np.empty((self.n_query,1))
        i = 0
        np.random.shuffle(self.classes)
        for C in self.classes:
            c_path = os.path.join(self.support_path,C)
            for f in os.listdir(c_path):
                file_path = os.path.join(c_path,f)
                #if self.augmentation:
                #X[i] = self.transformer.random_transform(img_to_array(load_img(file_path,target_size=self.target_size)))
                #else:
                X[i] = img_to_array(load_img(file_path,target_size=self.target_size))
                i += 1
        i = self.n_way
        lidx = 0
        for _ in range(self.n_query):
            cls_idx = np.random.randint(self.n_way)
            c = self.classes[cls_idx]
            c_path = os.path.join(self.query_path,str(c))
            qfile = os.listdir(c_path)[np.random.randint(len(os.listdir(c_path)))]
            #if self.augmentation:
            X[i] = self.transformer.random_transform(img_to_array(load_img(os.path.join(c_path,qfile),target_size=self.target_size)))
            #else:
                #X[i] = img_to_array(load_img(os.path.join(c_path,qfile),target_size=self.target_size))
            #X[i] = img_to_array(load_img(os.path.join(c_path,qfile),target_size=self.target_size))
            labels[lidx] = cls_idx
            i += 1
            lidx += 1
        label_query = to_categorical(labels,num_classes=self.n_way)

        return X[:self.n_way],X[self.n_way:],label_query