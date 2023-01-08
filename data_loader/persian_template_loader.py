import matplotlib.pyplot as plt
import os
from tensorflow import keras
import tensorflow as tf
from keras.preprocessing.image import load_img,img_to_array, ImageDataGenerator
import numpy as np
from tensorflow.keras.utils import to_categorical
import tensorflow
import os


class PERSIAN_Template_DataGenerator(tensorflow.keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(
        self,N,Ks,Kq,data_path='datasets\PERSIAN',
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
        self.augmentation = augmentation
        home = os.getcwd()
        #self.parent = os.path.join(self.current_path,os.pardir)
        #self.home = os.path.abspath(os.path.join(self.parent,os.pardir))
        original_path = os.path.join(home,self.data_path)
        #mini_imagenet_path = os.path.join(dataset_path,"GTSRB")
        #original_path = self.data_path
        template_path = os.path.join(original_path,"template")
        self.support_train_path = os.path.join(template_path,"seen")
        self.support_test_path = os.path.join(template_path,"unseen")
        self.query_train_path = os.path.join(original_path,"seen")
        self.query_test_path = os.path.join(original_path,"unseen")
        self.support_train_path_all = os.path.join(template_path,"all")
        self.query_train_path_all = os.path.join(original_path,"all")
        
        self.transformer = ImageDataGenerator(
            width_shift_range=0.1,
            height_shift_range=0.1,
            zoom_range=0.2,
            rotation_range=30,
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
        X_sample, X_query, label = self.__data_generation(self.N,self.Ks,self.Kq)
        #way = np.ones((self.way * self.shot, 1)) * self.way


        return [X_sample, X_query], label

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        pass

    def __data_generation(self,N,Ks,Kq):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        dataaugmentation = False
        train_classes = os.listdir(self.support_train_path)
        test_classes = os.listdir(self.support_test_path)
        all_classes = os.listdir(self.support_train_path_all)
        Dsupport = []
        Dquery = []
        label_query = []
        ql = 0.0
        if self.data_type == 'seen':
            classes = np.random.choice(train_classes,size=N,replace=False)
        elif self.data_type == 'all':
            classes = np.random.choice(all_classes,size=N,replace=False)
        else:
            classes = np.random.choice(test_classes,size=N,replace=False)
        
        for C in classes:
            if self.data_type == "seen":
                support_folders = os.path.join(self.support_train_path,C)
                query_folders = os.path.join(self.query_train_path,C)
            elif self.data_type == "all":
                support_folders = os.path.join(self.support_train_path_all,C)
                query_folders = os.path.join(self.query_train_path_all,C)
            else:
                support_folders = os.path.join(self.support_test_path,C)
                query_folders = os.path.join(self.query_test_path,C)
            
            support_samples = os.listdir(support_folders)
            query_samples = os.listdir(query_folders)
            idx_s = np.random.choice(len(support_samples),Ks,replace=False)
            idx_q = np.random.choice(len(query_samples),Kq,replace=False)
            data_s = np.array(support_samples)[idx_s]
            data_q = np.array(query_samples)[idx_q]
            X = []
            for file in data_s:
                file_path = os.path.join(support_folders,file)
                if self.augmentation:
                    X += [self.transformer.random_transform(img_to_array(load_img(file_path,target_size=self.target_size)))]
                else:
                    X += [img_to_array(load_img(file_path,target_size=self.target_size))]
            Dsupport += [X]
            X = []
            _label_q = []
            for file in data_q:
                file_path = os.path.join(query_folders,file)
                X += [img_to_array(load_img(file_path,target_size=self.target_size))]
                _label_q += [ql]
            ql += 1
            label_query += [_label_q]
            Dquery += [X]
        Dsupport = np.array(Dsupport)
        Dquery = np.array(Dquery)
        label_query = np.array(label_query)
        
        label_query = to_categorical(label_query.flatten(),num_classes=N)

        return Dsupport,Dquery,label_query