#####################
#                   #
#     Meta Test     #
#                   #
#####################

from tensorflow.keras.metrics import CategoricalAccuracy
import time
from data_loader import get_loader
from tensorflow.keras.models import load_model
from models.distances import Weighted_Euclidean_Distance
from models.stn import BilinearInterpolation,Localization
import tensorflow as tf
from tensorflow import keras




import argparse
parser = argparse.ArgumentParser('This is test')
parser.add_argument('--test',type = str,required = True,help = 'Test type')

args = parser.parse_args()

loader = get_loader(args.test) 
test_generator = loader.get_test_generator(batch=128,dim=64)



if __name__ == '__main__':
    model_h5 = 'model_files/best_models/densenet_' + args.test + '_whole.h5'
    #model_h5 = 'model_files/best_models/x_densenet_gtsrb2tt100k_whole.h5'
    loaded_model = load_model(
        model_h5,
        custom_objects={
            'Weighted_Euclidean_Distance':Weighted_Euclidean_Distance,
            'BilinearInterpolation':BilinearInterpolation,
            'Localization':Localization},compile=False)
    
    metric = CategoricalAccuracy()
    loaded_model.compile(metrics=metric)
     
    best_acc = 0
    start_time = time.time()
    for iter in range(1):
        print(f'\033[0;36mstrating test iteration {iter+1}\033[0m')
        _,acc = loaded_model.evaluate(test_generator,verbose=1)
        if acc > best_acc:
            best_acc = acc
    print('\033[0;31m')
    print('+---------------------------+')
    print('|       Testing Report      |')
    print('+---------------------------+')
    print(f'|      Test    |{args.test}|')
    print('+---------------------------+')
    print(f'|Best accuracy |{best_acc:.4f}     |')
    print('+---------------------------+')
    print('\033[0m')
