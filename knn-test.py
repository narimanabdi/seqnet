#####################
#                   #
#     Meta Test     #
#                   #
#####################

from tensorflow.keras.metrics import CategoricalAccuracy
from time import time
from data_loader import get_loader
from tensorflow.keras.models import load_model
from models.stn import BilinearInterpolation,Localization
from sklearn.neighbors import KNeighborsClassifier
from tensorflow.keras.utils import to_categorical
import numpy as np
import tensorflow as tf
from tensorflow.keras.metrics import Mean




import argparse
parser = argparse.ArgumentParser('This is test')
parser.add_argument('--test',type = str,default='gtsrb2tt100k',help = 'Test type')
parser.add_argument('--batch',type = int,default=8)

args = parser.parse_args()

loader = get_loader(args.test) 
batch = args.batch
test_generator = loader.get_test_generator(batch=batch,dim=64,type='unseen')

acc_tracker = Mean(name='KNN_Accuracy')





if __name__ == '__main__':
    
    encoder_h5 = 'model_files/best_encoders/densenet_' + args.test + '_encoder.h5'
    #encoder_h5 = 'model_files/best_encoders/w_densenet_gtsrb2tt100k_encoder.h5'
    loaded_encoder = load_model(
        encoder_h5,
        custom_objects={
            'BilinearInterpolation':BilinearInterpolation,
            'Localization':Localization},compile=False)

    t = iter(test_generator)
    [Xs,Xq],y = next(t)

    

    Zs = loaded_encoder(Xs)
    n_cls = len(Xs)
    knn = KNeighborsClassifier(n_neighbors=1,n_jobs=-1,metric='l2')
    y_train = to_categorical(np.arange(n_cls),num_classes=n_cls)
    knn.fit(Zs,y_train);

    #calculate FPS
    start = time()
    Zq = loaded_encoder(Xq[0:1])
    acc = knn.score(Zq,y[0])
    time_per_image = time() - start
    fps = int(1/time_per_image)
    print(time_per_image)
    
    
    acc = 0
    start = time()
    pb = tf.keras.utils.Progbar(len(test_generator),verbose=1)
    for i,z in enumerate(test_generator):
        [Xs,Xq],y_test = z
        Zq = loaded_encoder(Xq)
        #acc += knn.score(Zq,y_test)
        acc = knn.score(Zq,y_test)
        acc_tracker.update_state(acc)
        #values=[('knn accuracy',acc_tracker.result())]
        pb.add(1)#,values = values)
    end = time() - start

    #time_per_image = end / (len(test_generator) * batch)
    #fps = int(1/time_per_image)
    #mean_accuracy = np.round((acc / len(test_generator))*100,2)

    print('\033[0;31m')
    print('+---------------------------+')
    print('|    1-NN Testing Report    |')
    print('+---------------------------+')
    print(f'|      Test    |{args.test}|')
    print('+---------------------------+')
    print(f'|Mean accuracy |{acc_tracker.result():.4f}     |')
    print('+---------------------------+')
    print('+---------------------------+')
    print(f'|FPS |{fps}     |')
    print('+---------------------------+')
    print('+---------------------------+')
    print(f'|Inference time per image |{time_per_image:.4f}     |')
    print('+---------------------------+')
    print('\033[0m')
