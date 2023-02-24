#########################
#                       #
# Nearest Neighbor Test #
#                       #
#########################

from time import time
from data_loader import get_loader
from tensorflow.keras.models import load_model
from models.stn import BilinearInterpolation,Localization
from sklearn.neighbors import KNeighborsClassifier
from tensorflow.keras.utils import to_categorical
import numpy as np
import tensorflow as tf
from tensorflow.keras.metrics import Mean
from models.metrics import accuracy

import argparse
parser = argparse.ArgumentParser('Nearest Neighbor Accuracy')
parser.add_argument('--test',type = str,default='gtsrb2tt100k',help = 'Test type')
parser.add_argument('--mode',type = str,default='base')
parser.add_argument('--batch',type = int,default=128)


args = parser.parse_args()

loader = get_loader(args.test) 
batch = args.batch
test_generator = loader.get_test_generator(batch=batch,dim=64)

acc_tracker = Mean(name='Nearest_Neighbor_Accuracy')
time_tracker = Mean(name='Time')

if __name__ == '__main__':
    if args.mode == 'base':
        encoder_h5 = 'model_files/best_encoders/densenet_' +\
              args.test + '_encoder.h5'
    if args.mode == 'mini':
        encoder_h5 = 'model_files/best_encoders/densenetmini_' +\
              args.test + '_encoder.h5'
    if args.mode == 'random':
        encoder_h5 = 'model_files/best_encoders/densenet_' +\
              args.test + '_encoder_random.h5'
    if args.mode == 'resnet':
        encoder_h5 = 'model_files/best_encoders/resnet_' +\
              args.test + '_encoder.h5'
    if args.mode == 'mobilenet':
        encoder_h5 = 'model_files/best_encoders/mobilenet_' +\
              args.test + '_encoder.h5'
    
    loaded_encoder = load_model(
        encoder_h5,
        custom_objects={
            'BilinearInterpolation':BilinearInterpolation,
            'Localization':Localization},compile=False)

    t = iter(test_generator)
    [Xs,Xq],y = next(t)

    Zs = loaded_encoder(Xs)
    n_cls = len(Xs)
    nn = KNeighborsClassifier(n_neighbors=1,n_jobs=-1,metric='l2')
    y_train = to_categorical(np.arange(n_cls),num_classes=n_cls)
    nn.fit(Zs,y_train);

    print('\033[3;34mStart Timing Benchmark\033[0m')

    
    for _,x in enumerate(Xq):
        start = time()
        zq = loaded_encoder(tf.expand_dims(x,axis=0))
        y_pred = nn.predict(zq)
        time_tracker.update_state(time() - start)
    time_per_image = time_tracker.result()
    fps = 1/time_per_image
    
    print('\033[0;32mStart Nearest Neighbor Test\033[0m')
    best_acc = 0
    for i,data in enumerate(test_generator):
        [Xs,Xq],y_test = data
        Zq = loaded_encoder(Xq)
        acc_tracker.update_state(nn.score(Zq,y_test)) 
    #if acc_tracker.result() > best_acc:
        #best_acc = acc_tracker.result()
    #acc_tracker.reset_state()

    print('\033[0;31m')
    print('+---------------------------+')
    print('|    1-NN Testing Report    |')
    print('+---------------------------+')
    print(f'|      Test    |{args.test}|')
    print('+---------------------------+')
    print(f'|Mean accuracy |{acc_tracker.result():.4f}|')
    print('+---------------------------+')
    print('+---------------------------+')
    print(f'|Average FPS |{fps:.4f}     |')
    print('+---------------------------+')
    print('+---------------------------+')
    print(f'|Inference time per image |{time_per_image:.2f}|')
    print('+---------------------------+')
    print('\033[0m')
