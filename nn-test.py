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

from prettytable import PrettyTable
from utils import count_params

import argparse
parser = argparse.ArgumentParser('Nearest Neighbor Test')
parser.add_argument('--test',type = str,default='gtsrb2tt100k',help = 'Test type')
parser.add_argument('--mode',type = str,default='base')
parser.add_argument('--batch',type = int,default=128)
parser.add_argument('--metric',default='l2')


args = parser.parse_args()

loader = get_loader(args.test) 
batch = args.batch
test_generator = loader.get_test_generator(batch=batch,dim=64,shuffle=False)

acc_tracker = Mean(name='Nearest_Neighbor_Accuracy')
time_tracker = Mean(name='Time')

def calc_fps(encoder_h5,test='gtsrb2tt100k'):
    loaded_encoder = load_model(
        encoder_h5,
        custom_objects={
            'BilinearInterpolation':BilinearInterpolation,
            'Localization':Localization},compile=False)

    loader = get_loader(test) 
    test_generator = loader.get_test_generator(batch=batch,dim=64,shuffle=False)
    t = iter(test_generator)
    [Xs,Xq],y = next(t)

    Zs = loaded_encoder(Xs)
    n_cls = len(Xs)
    nn = KNeighborsClassifier(n_neighbors=1,n_jobs=-1,metric = args.metric)
    y_train = to_categorical(np.arange(n_cls),num_classes=n_cls)
    nn.fit(Zs,y_train);

    print('\033[3;34mStart Timing Benchmark\033[0m')
    fps = 0
    for i in range(2):
        for _,x in enumerate(Xq):
            start = time()
            zq = loaded_encoder(tf.expand_dims(x,axis=0))
            y_pred = nn.predict(zq)
            time_tracker.update_state(time() - start)
        time_per_image = time_tracker.result()
        fps += 1/time_per_image
    fps = fps / 2    
    myTable = PrettyTable([" 1-NN Testing Report", "VALUES"])
    myTable = PrettyTable([" 1-NN Testing Report", "VALUES"])
    myTable.add_row(["Test", f'{args.test}'])
    myTable.add_row(["Average FPS", f'{fps:.1f}'])
    #myTable.add_row(["Inference time per image", f'{time_per_image:.2f}s'])
    myTable.add_row(["Model Parameters", f'{count_params(loaded_encoder):.2f}M'])
    print('\033[0;31m')
    print(myTable)
    print('\033[0m')

def calc_accuracy(encoder_h5,test='gtsrb2tt100k', mode = 'standard'):
    
    loaded_encoder = load_model(
        encoder_h5,
        custom_objects={
            'BilinearInterpolation':BilinearInterpolation,
            'Localization':Localization},compile=False)
    loader = get_loader(test) 
    test_generator = loader.get_test_generator(batch=batch,dim=64,shuffle=False)
    t = iter(test_generator)
    [Xs,Xq],y = next(t)

    Zs = loaded_encoder(Xs)
    n_cls = len(Xs)
    nn = KNeighborsClassifier(n_neighbors=1,n_jobs=-1,metric = args.metric)
    y_train = to_categorical(np.arange(n_cls),num_classes=n_cls)
    nn.fit(Zs,y_train);
    
    print('\033[0;32mStart Nearest Neighbor Test\033[0m')
    best_acc = 0
    for i,data in enumerate(test_generator):
        [Xs,Xq],y_test = data
        Zq = loaded_encoder(Xq)
        acc_tracker.update_state(nn.score(Zq,y_test)) 
    #if acc_tracker.result() > best_acc:
        #best_acc = acc_tracker.result()
    #acc_tracker.reset_state()
    myTable = PrettyTable([" 1-NN Testing Report", ""])
    myTable.add_row(["Evaluation", test])
    myTable.add_row(["Test Mode", mode])
    myTable.add_row(["Mean Accuracy", f'{acc_tracker.result()*100.0:.2f}'])
    #myTable.add_row(["Inference time per image", f'{time_per_image:.2f}s'])
    myTable.add_row(["Model Parameters", f'{count_params(loaded_encoder):.2f}M'])
    print('\033[0;31m')
    print(myTable)
    print('\033[0m')
    

def all_test():
    calc_accuracy('model_files/best_encoders/densenet_gtsrb2tt100k_encoder.h5', test='gtsrb2tt100k')
    acc_tracker.reset_state()
    calc_accuracy('model_files/best_encoders/densenet_gtsrb_encoder.h5', test='gtsrb')
    acc_tracker.reset_state()
    calc_accuracy('model_files/best_encoders/densenet_belga2flick_encoder.h5', test='belga2flick')
    acc_tracker.reset_state()
    calc_accuracy('model_files/best_encoders/densenet_belga2toplogo_encoder.h5', test='belga2toplogo')
    acc_tracker.reset_state()
    calc_accuracy('model_files/best_encoders/densenet_gtsrb2flick_encoder.h5', test='gtsrb2flick')
    acc_tracker.reset_state()
    calc_accuracy('model_files/best_encoders/densenet_gtsrb2toplogo_encoder.h5', test='gtsrb2toplogo')
    acc_tracker.reset_state()
    calc_accuracy('model_files/best_encoders/densenetmini_gtsrb2tt100k_encoder.h5', mode = 'mini')
    acc_tracker.reset_state()
    calc_accuracy('model_files/best_encoders/densenet_gtsrb2tt100k_encoder_random.h5', mode = 'random')
    acc_tracker.reset_state()
    calc_accuracy('model_files/best_encoders/resnet_gtsrb2tt100k_encoder.h5', mode = 'resnet')
    acc_tracker.reset_state()
    calc_accuracy('model_files/best_encoders/mobilenet_gtsrb2tt100k_encoder.h5', mode = 'mobilenet')
    acc_tracker.reset_state()
    calc_accuracy('model_files/best_encoders/densenet_gtsrb2tt100k_encoder_wo_stn.h5', mode = 'without STN')
    acc_tracker.reset_state()

if __name__ == '__main__':
    if args.mode == 'base':
        encoder_h5 = 'model_files/best_encoders/densenet_gtsrb2tt100k_encoder_wo_stn.h5'
        calc_accuracy(encoder_h5, test=args.test)
    if args.mode == 'mini':
        encoder_h5 = 'model_files/best_encoders/densenetmini_' +\
              args.test + '_encoder.h5'
        calc_accuracy(encoder_h5, test=args.test)
    if args.mode == 'random':
        encoder_h5 = 'model_files/best_encoders/densenet_' +\
              args.test + '_encoder_random.h5'
        calc_accuracy(encoder_h5, test=args.test)
    if args.mode == 'resnet':
        encoder_h5 = 'model_files/best_encoders/resnet_' +\
              args.test + '_encoder.h5'
        calc_accuracy(encoder_h5, test=args.test)
    if args.mode == 'mobilenet':
        encoder_h5 = 'model_files/best_encoders/mobilenet_' +\
              args.test + '_encoder.h5'
        calc_accuracy(encoder_h5, test=args.test)
    if args.mode == 'no-stn':
        encoder_h5 = 'model_files/best_encoders/densenet_gtsrb2tt100k_encoder_wo_stn.h5'
        calc_accuracy(encoder_h5, test=args.test)
    if args.mode == 'fps':
        encoder_h5 = 'model_files/best_encoders/densenet_gtsrb2tt100k_encoder.h5'
        #calc_accuracy(encoder_h5, test=args.test)
        calc_fps(encoder_h5='model_files/best_encoders/densenet_gtsrb2tt100k_encoder_wo_stn.h5')
    if args.mode == 'all':
        all_test()
    