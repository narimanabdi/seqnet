from tensorflow.keras.preprocessing.image import load_img,img_to_array
import numpy as np
from tensorflow.keras.utils import to_categorical
import tensorflow as tf
import os

def standardize(img):
    mean = np.mean(img)
    std = np.std(img)
    return (img - mean) / std


class GTSRB_Generator(tf.keras.utils.Sequence):
    """
    Generating Batch of Data
    n_way: number of ways (classes)
    k_shot: number of shots
    batch: batch of data (query samples)
    """
    def __init__(
        self,n_way,k_shot,data_path='datasets/GTSRB',
        batch=64,data_type = 'all',target_size=(64,64),shuffle=True):
        #Initialization
        self.n_way = n_way
        self.k_shot = k_shot
        self.data_path = data_path
        self.batch = batch
        self.data_type = data_type
        self.target_size = target_size
        self.standardize = standardize
        self.shuffle = shuffle

        
        home = os.getcwd()
        original_path = os.path.join(home,self.data_path)
        template_path = os.path.join(original_path,"template")

        if self.data_type == 'seen':
            self.support_path = os.path.join(template_path,"seen")
            self.query_path = os.path.join(original_path,"seen")
        elif self.data_type == 'all':
            self.support_path = os.path.join(template_path,"all")
            self.query_path = os.path.join(original_path,"all")
        else:
            self.support_path = os.path.join(template_path,"unseen")
            self.query_path = os.path.join(original_path,"unseen")

        self.classes = np.asarray(os.listdir(self.query_path))
        self.classes.sort()
   
        self.IDs = []
        self.labels_IDs = []
        for i,C in enumerate(self.classes):
            c_path = os.path.join(self.query_path,C)
            for f in os.listdir(c_path):
                self.IDs += [C+'/'+f]
                self.labels_IDs += [i]
        self.labels_IDs = np.asarray(self.labels_IDs)
        
        self.on_epoch_end()
    
    def __len__(self):
        'Denotes the number of batches per epoch'
        return (len(self.labels_IDs) // self.batch)


    def __getitem__(self,index):
        'Generate one batch of data'
        # Generate data
        indexes = self.indexes[index*self.batch:(index+1)*self.batch]
        X_sample, X_query, label = self.__data_generation(indexes)
        return [X_sample, X_query], label

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.labels_IDs))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __data_generation(self,indexes):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        
        X = np.empty((self.n_way * self.k_shot + self.batch,*self.target_size,3))
        y = np.empty((self.batch,1))

        for i,C in enumerate(self.classes):
            c_path = os.path.join(self.support_path,C)
            for f in os.listdir(c_path):
                file_path = os.path.join(c_path,f)
                #X[i] = img_to_array(load_img(file_path,target_size=self.target_size))
     
                X[i] = self.standardize(img_to_array(load_img(
                    file_path,target_size=self.target_size,
                    interpolation='bilinear')))
        
        for i,idx in enumerate(indexes):
            file_path = os.path.join(self.query_path,self.IDs[idx])
            X[i+self.n_way] = self.standardize(img_to_array(load_img(
                    file_path,target_size=self.target_size,
                    interpolation='bilinear')))
            y[i] = self.labels_IDs[idx]
        label_query = to_categorical(y,num_classes=self.n_way)

        return tf.constant(X[:self.n_way],dtype='float32'),\
            tf.constant(X[self.n_way:],dtype='float32'),\
                tf.constant(label_query,dtype='float32')

class TT100K_Generator(tf.keras.utils.Sequence):
    """
    Generating Batch of Data
    n_way: number of ways (classes)
    k_shot: number of shots
    batch: batch of data (query samples)
    """
    def __init__(
        self,n_way,k_shot,data_path='datasets/TT100K',
        batch=64,data_type = 'all',target_size=(64,64),shuffle=True):
        #Initialization
        self.n_way = n_way
        self.k_shot = k_shot
        self.data_path = data_path
        self.batch = batch
        self.data_type = data_type
        self.target_size = target_size
        self.standardize = standardize
        self.shuffle = shuffle
        
        home = os.getcwd()
        original_path = os.path.join(home,self.data_path)
        template_path = os.path.join(original_path,"template")

        if self.data_type == 'seen':
            self.support_path = os.path.join(template_path,"seen")
            self.query_path = os.path.join(original_path,"seen")
        elif self.data_type == 'all':
            self.support_path = os.path.join(template_path,"all")
            self.query_path = os.path.join(original_path,"all")
        elif self.data_type == 'FT':
            self.support_path = os.path.join(template_path,"all")
            self.query_path = os.path.join(original_path,"FT")
        else:
            self.support_path = os.path.join(template_path,"unseen")
            self.query_path = os.path.join(original_path,"unseen")


        
        self.classes = np.asarray(os.listdir(self.query_path))
       
        self.classes.sort()
   
        self.IDs = []
        self.labels_IDs = []
        for i,C in enumerate(self.classes):
            c_path = os.path.join(self.query_path,C)
            for f in os.listdir(c_path):
                self.IDs += [C+'/'+f]
                self.labels_IDs += [i]
        self.labels_IDs = np.asarray(self.labels_IDs)
        #self.indexes = np.arange(len(self.labels_IDs))
        #np.random.shuffle(self.indexes)
        
        self.on_epoch_end()
    
    def __len__(self):
        'Denotes the number of batches per epoch'
        return (len(self.labels_IDs) // self.batch)


    def __getitem__(self,index):
        'Generate one batch of data'
        # Generate data
        indexes = self.indexes[index*self.batch:(index+1)*self.batch]
        X_sample, X_query, label = self.__data_generation(indexes)
        return [X_sample, X_query], label

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.labels_IDs))
        if self.shuffle:
            np.random.shuffle(self.indexes)
        else:
            pass

    def __data_generation(self,indexes):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        
        X = np.empty((self.n_way * self.k_shot + self.batch,*self.target_size,3))
        y = np.empty((self.batch,1))
     
        for i,C in enumerate(self.classes):
            c_path = os.path.join(self.support_path,C)
            for f in os.listdir(c_path):
                file_path = os.path.join(c_path,f)
                X[i] = self.standardize(img_to_array(load_img(
                    file_path,target_size=self.target_size,
                    interpolation='bilinear')))
        
        for i,idx in enumerate(indexes):
            file_path = os.path.join(self.query_path,self.IDs[idx])
            X[i+self.n_way] = self.standardize(img_to_array(load_img(
                    file_path,target_size=self.target_size,
                    interpolation='bilinear')))
            y[i] = self.labels_IDs[idx]
        label_query = to_categorical(y,num_classes=self.n_way)


        return tf.constant(X[:self.n_way],dtype='float32'),\
            tf.constant(X[self.n_way:],dtype='float32'),\
                tf.constant(label_query,dtype='float32')

class FLICKR32_Generator(tf.keras.utils.Sequence):
    """
    Generating Batch of Data
    n_way: number of ways (classes)
    k_shot: number of shots
    batch: batch of data (query samples)
    """
    def __init__(
        self,n_way,k_shot,data_path='datasets/flickr32',
        batch=64,data_type = 'all',augmentation=False, shuffle = True,
        target_size=(64,64)):
        #Initialization
        self.n_way = n_way
        self.k_shot = k_shot
        self.data_path = data_path
        self.batch = batch
        self.data_type = data_type
        self.target_size = target_size
        self.augmentation = augmentation
        self.standardize = standardize
        self.shuffle = shuffle
        
        home = os.getcwd()
        original_path = os.path.join(home,self.data_path)
        
        
        self.support_path = os.path.join(original_path,"template")
        if data_type == 'all':
            self.query_path = os.path.join(original_path,"all")
        else:
            self.query_path = os.path.join(original_path,"FT")
        self.classes = np.asarray(os.listdir(self.support_path))
        self.classes.sort()
    
        self.IDs = []
        self.labels_IDs = []
        for i,C in enumerate(self.classes):
            c_path = os.path.join(self.query_path,C)
            for f in os.listdir(c_path):
                self.IDs += [C+'/'+f]
                self.labels_IDs += [i]
        self.labels_IDs = np.asarray(self.labels_IDs)
        
        self.on_epoch_end()
    
    def __len__(self):
        'Denotes the number of batches per epoch'
        return (len(self.labels_IDs) // self.batch)


    def __getitem__(self,index):
        'Generate one batch of data'
        # Generate data
        indexes = self.indexes[index*self.batch:(index+1)*self.batch]
        X_sample, X_query, label = self.__data_generation(indexes)
        return [X_sample, X_query], label

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.labels_IDs))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __data_generation(self,indexes):
        'Generates data containing batch_size samples' 
        # Initialization
        
        X = np.empty((self.n_way * self.k_shot + self.batch,*self.target_size,3))
        y = np.empty((self.batch,1))
     
        for i,C in enumerate(self.classes):
            c_path = os.path.join(self.support_path,C)
            for f in os.listdir(c_path):
                file_path = os.path.join(c_path,f)
                X[i] = self.standardize(img_to_array(load_img(
                    file_path,target_size=self.target_size,
                    interpolation='bilinear'))) 
        
        for i,idx in enumerate(indexes):
            file_path = os.path.join(self.query_path,self.IDs[idx])
            X[i+self.n_way] = self.standardize(img_to_array(load_img(
                    file_path,target_size=self.target_size,
                    interpolation='bilinear')))
            y[i] = self.labels_IDs[idx]
        label_query = to_categorical(y,num_classes=self.n_way)


        return tf.constant(X[:self.n_way],dtype='float32'),\
            tf.constant(X[self.n_way:],dtype='float32'),\
                tf.constant(label_query,dtype='float32')

class BELGA_Generator(tf.keras.utils.Sequence):
    """
    Generating Batch of Data
    n_way: number of ways (classes)
    k_shot: number of shots
    batch: batch of data (query samples)
    """
    def __init__(
        self,n_way,k_shot,data_path='datasets/belga',
        batch=64,data_type = 'all', shuffle = True,
        target_size=(64,64)):
        #Initialization
        self.n_way = n_way
        self.k_shot = k_shot
        self.data_path = data_path
        self.batch = batch
        self.data_type = data_type
        self.target_size = target_size
        self.shuffle = shuffle
        
        home = os.getcwd()
        original_path = os.path.join(home,self.data_path)
        self.support_path = os.path.join(original_path,"template")
        if data_type == 'all':
            self.query_path = os.path.join(original_path,"all")
        else:
            self.query_path = os.path.join(original_path,"FT")
        self.classes = np.asarray(os.listdir(self.support_path))
        self.classes.sort()

        self.standardize = standardize

        self.IDs = []
        self.labels_IDs = []
        for i,C in enumerate(self.classes):
            c_path = os.path.join(self.query_path,C)
            for f in os.listdir(c_path):
                self.IDs += [C+'/'+f]
                self.labels_IDs += [i]
        self.labels_IDs = np.asarray(self.labels_IDs)
        
        
        self.on_epoch_end()
    
    def __len__(self):
        'Denotes the number of batches per epoch'
        return (len(self.labels_IDs) // self.batch)


    def __getitem__(self,index):
        'Generate one batch of data'
        # Generate data
        indexes = self.indexes[index*self.batch:(index+1)*self.batch]
        X_sample, X_query, label = self.__data_generation(indexes)
        return [X_sample, X_query], label

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.labels_IDs))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __data_generation(self,indexes):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        
        X = np.empty((self.n_way * self.k_shot + self.batch,*self.target_size,3))
        y = np.empty((self.batch,1))
     
        for i,C in enumerate(self.classes):
            c_path = os.path.join(self.support_path,C)
            for f in os.listdir(c_path):
                file_path = os.path.join(c_path,f)
                X[i] = self.standardize(img_to_array(load_img(
                    file_path,target_size=self.target_size,
                    interpolation='bilinear')))
        
        for i,idx in enumerate(indexes):
            file_path = os.path.join(self.query_path,self.IDs[idx])
            X[i+self.n_way] = self.standardize(img_to_array(load_img(
                    file_path,target_size=self.target_size,
                    interpolation='bilinear')))
            y[i] = self.labels_IDs[idx]
        label_query = to_categorical(y,num_classes=self.n_way)


        return tf.constant(X[:self.n_way],dtype='float32'),\
            tf.constant(X[self.n_way:],dtype='float32'),\
                tf.constant(label_query,dtype='float32')

class TOPLOGO10_Generator(tf.keras.utils.Sequence):
    """
    Generating Batch of Data
    n_way: number of ways (classes)
    k_shot: number of shots
    batch: batch of data (query samples)
    """
    def __init__(
        self,n_way,k_shot,data_path='datasets/toplogo10',
        batch=64,data_type = 'all',shuffle=True,
        target_size=(64,64)):
        #Initialization
        self.n_way = n_way
        self.k_shot = k_shot
        self.data_path = data_path
        self.batch = batch
        self.data_type = data_type
        self.target_size = target_size
        self.shuffle = shuffle
        
        home = os.getcwd()
        original_path = os.path.join(home,self.data_path)
        
        #add transformer for data augmentation
        self.standardize = standardize

        self.support_path = os.path.join(original_path,"template")
        self.query_path = os.path.join(original_path,"all")
        self.classes = np.asarray(os.listdir(self.support_path))
        self.classes.sort()
     
        self.IDs = []
        self.labels_IDs = []
        for i,C in enumerate(self.classes):
            c_path = os.path.join(self.query_path,C)
            for f in os.listdir(c_path):
                self.IDs += [C+'/'+f]
                self.labels_IDs += [i]
        self.labels_IDs = np.asarray(self.labels_IDs)
        
        self.on_epoch_end()
    
    def __len__(self):
        'Denotes the number of batches per epoch'
        return (len(self.labels_IDs) // self.batch)


    def __getitem__(self,index):
        'Generate one batch of data'
        # Generate data
        indexes = self.indexes[index*self.batch:(index+1)*self.batch]
        X_sample, X_query, label = self.__data_generation(indexes)
        return [X_sample, X_query], label

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.labels_IDs))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __data_generation(self,indexes):
        'Generates data containing batch_size samples'
        # Initialization
        
        X = np.empty((self.n_way * self.k_shot + self.batch,*self.target_size,3))
        y = np.empty((self.batch,1))
     
        for i,C in enumerate(self.classes):
            c_path = os.path.join(self.support_path,C)
            for f in os.listdir(c_path):
                file_path = os.path.join(c_path,f)
                X[i] = self.standardize(img_to_array(load_img(
                    file_path,target_size=self.target_size,
                    interpolation='bilinear'))) 
        
        for i,idx in enumerate(indexes):
            file_path = os.path.join(self.query_path,self.IDs[idx])
            X[i+self.n_way] = self.standardize(img_to_array(load_img(
                    file_path,target_size=self.target_size,
                    interpolation='bilinear'))) 
            y[i] = self.labels_IDs[idx]
        label_query = to_categorical(y,num_classes=self.n_way)


        return tf.constant(X[:self.n_way],dtype='float32'),\
            tf.constant(X[self.n_way:],dtype='float32'),\
                tf.constant(label_query,dtype='float32')
    
class GTSRB_Generator(tf.keras.utils.Sequence):
    """
    Generating Batch of Data
    n_way: number of ways (classes)
    k_shot: number of shots
    batch: batch of data (query samples)
    """
    def __init__(
        self,n_way,k_shot,data_path='datasets/GTSRB',
        batch=64,data_type = 'all',target_size=(64,64),shuffle=True):
        #Initialization
        self.n_way = n_way
        self.k_shot = k_shot
        self.data_path = data_path
        self.batch = batch
        self.data_type = data_type
        self.target_size = target_size
        self.standardize = standardize
        self.shuffle = shuffle

        
        home = os.getcwd()
        original_path = os.path.join(home,self.data_path)
        template_path = os.path.join(original_path,"template")

        if self.data_type == 'seen':
            self.support_path = os.path.join(template_path,"seen")
            self.query_path = os.path.join(original_path,"seen")
        elif self.data_type == 'all':
            self.support_path = os.path.join(template_path,"all")
            self.query_path = os.path.join(original_path,"all")
        else:
            self.support_path = os.path.join(template_path,"unseen")
            self.query_path = os.path.join(original_path,"unseen")

        self.classes = np.asarray(os.listdir(self.query_path))
        self.classes.sort()
   
        self.IDs = []
        self.labels_IDs = []
        for i,C in enumerate(self.classes):
            c_path = os.path.join(self.query_path,C)
            for f in os.listdir(c_path):
                self.IDs += [C+'/'+f]
                self.labels_IDs += [i]
        self.labels_IDs = np.asarray(self.labels_IDs)
        
        self.on_epoch_end()
    
    def __len__(self):
        'Denotes the number of batches per epoch'
        return (len(self.labels_IDs) // self.batch)


    def __getitem__(self,index):
        'Generate one batch of data'
        # Generate data
        indexes = self.indexes[index*self.batch:(index+1)*self.batch]
        X_sample, X_query, label = self.__data_generation(indexes)
        return [X_sample, X_query], label

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.labels_IDs))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __data_generation(self,indexes):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        
        X = np.empty((self.n_way * self.k_shot + self.batch,*self.target_size,3))
        y = np.empty((self.batch,1))

        for i,C in enumerate(self.classes):
            c_path = os.path.join(self.support_path,C)
            for f in os.listdir(c_path):
                file_path = os.path.join(c_path,f)
                #X[i] = img_to_array(load_img(file_path,target_size=self.target_size))
     
                X[i] = self.standardize(img_to_array(load_img(
                    file_path,target_size=self.target_size,
                    interpolation='bilinear')))
        
        for i,idx in enumerate(indexes):
            file_path = os.path.join(self.query_path,self.IDs[idx])
            X[i+self.n_way] = self.standardize(img_to_array(load_img(
                    file_path,target_size=self.target_size,
                    interpolation='bilinear')))
            y[i] = self.labels_IDs[idx]
        label_query = to_categorical(y,num_classes=self.n_way)

        return tf.constant(X[:self.n_way],dtype='float32'),\
            tf.constant(X[self.n_way:],dtype='float32'),\
                tf.constant(label_query,dtype='float32')
class MINI_Generator(tf.keras.utils.Sequence):
    """
    Generating Batch of Data
    n_way: number of ways (classes)
    k_shot: number of shots
    batch: batch of data (query samples)
    """
    def __init__(
        self,n_way,k_shot,qnum,data_path='datasets/mini',
        data_type = 'train',episode = 1000,
        target_size=(64,64)):
        #Initialization
        self.n_way = n_way
        self.k_shot = k_shot
        self.data_path = data_path
        self.qnum = qnum
        self.data_type = data_type
        self.target_size = target_size
        self.episode = episode
        
        home = os.getcwd()
        original_path = os.path.join(home,self.data_path)
   
        
        #add transformer for data augmentation

        if self.data_type == 'train':
            self.path = os.path.join(original_path,"images_background")
     
        elif self.data_type == 'test':
            self.path = os.path.join(original_path,"images_evaluation")
  


        self.classes = np.asarray(os.listdir(self.path))
   
        print(f'Loading {self.data_type} data')
        
        Bq = []
        Yq = []
        
        for i,C in enumerate(self.classes):
            c_path = os.path.join(self.path,C)
            Dq = []
            labels = []
            for f in os.listdir(c_path):
                file_path = os.path.join(c_path,f)
                #Dq += [img_to_array(load_img(file_path,target_size=self.target_size))]
                Dq += [file_path]
                labels += [i]
            Bq += [Dq]
            Yq += [labels]
        self.data = np.asarray(Bq)
        self.labels = np.asarray(Yq)
        
        self.on_epoch_end()
    
    def __len__(self):
        'Denotes the number of batches per epoch'
        return self.episode


    def __getitem__(self,index):
        'Generate one batch of data'
        # Generate data
        #self.classes = np.random.choice(range(self.data.shape[0]),size=self.n_way,replace=False)
        choices = np.random.permutation(list(set(range(self.data.shape[0]))))
        #self.classes = np.random.choice(range(self.data.shape[0]),size=self.n_way,replace=False)
        self.classes = choices[:self.n_way]
        X_sample, X_query, label = self.__data_generation(self.classes)
        return [X_sample, X_query], label

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        pass
        #self.classes = np.random.choice(range(self.data.shape[0]),size=self.n_way,replace=False)

    def __data_generation(self,classes):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        
        X = np.empty((self.n_way*(1+self.qnum),*self.target_size,3))#20
        y = np.empty((self.n_way*self.qnum,1))#15

        for i,idx in enumerate(classes):
            file_path = self.data[idx][np.random.randint(600)]
            X[i] = img_to_array(load_img(file_path,target_size=self.target_size))
            
        
        for i in range(self.n_way*self.qnum):
            idx = np.random.choice(classes)
            file_path = self.data[idx][np.random.randint(600)]
            X[i+self.n_way] = img_to_array(load_img(file_path,target_size=self.target_size))#self.data[idx][np.random.randint(600)]
            y[i] = np.where(classes == idx)[0][0]
        label_query = to_categorical(y,num_classes=self.n_way)

        return X[:self.n_way],X[self.n_way:],label_query