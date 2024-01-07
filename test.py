###############################
#                             #
# Nearest Neighbor Evaluation #
#                             #
###############################

from time import time
from data_loader import get_loader
from tensorflow.keras.models import load_model
from models.stn import BilinearInterpolation,Localization
import numpy as np
import tensorflow as tf
from tensorflow.keras.metrics import Mean
from prettytable import PrettyTable
from models.utils import count_params
import argparse
from models.distances import Cosine_Distance, Euclidean_Distance
from keras.metrics import Recall, Precision
import json

#read config
json_file = open('cfgs/test.json')
config = json.load(json_file)
encoder_file = 'model_files/best_encoders/' + config['backbone'] + '_' + config['test'] + '_encoder.h5'
device = config['device']
distance_metric = config['distance']
batch_size = config['batch_size']

if distance_metric == 'euclidean':
    dist_fn = Euclidean_Distance()
elif distance_metric == 'cosine':
    dist_fn = Cosine_Distance()
#metric tracker
acc_tracker = Mean(name='Nearest_Neighbor_Accuracy')
rec_tracker = Mean(name='Nearest_Neighbor_Recall')
pre_tracker = Mean(name='Nearest_Neighbor_Precision')
_recall = Recall()
_precision = Precision()

time_tracker = Mean(name='Time')
@tf.function
def nn(model,inp,ztemplates):
    z = model(tf.expand_dims(inp,axis=0))
    return dist_fn([ztemplates,z])

def run_test(encoder_h5,data,batch):
    loaded_encoder = load_model(
        encoder_h5,
        custom_objects={
            'BilinearInterpolation':BilinearInterpolation,
            'Localization':Localization},compile=False)
    loader = get_loader(data) 
    print(loader)
    test_generator = loader.get_test_generator(batch=batch,dim=64,shuffle=False)
    t = iter(test_generator)
    [Xs,_Xq],_y = next(t)
    del _y,_Xq,t
    Zs = loaded_encoder(Xs)
    del Xs
    print('\033[0;32mStart Nearest Neighbor Evaluation\033[0m')
    #pb = tf.keras.utils.Progbar(len(test_generator),verbose=1)
    tval = []
    fpsval = []
    ITER = len(test_generator)
    for iteration,(d,y_test) in enumerate(test_generator):
        preds = []
        targets = []
        for i,x in enumerate(d[1]):
            s = time()
            p = nn(loaded_encoder,x,Zs)
            tval.append(time() - s)
            fpsval.append(1.0 / tval[-1])
            #we use argmax because we use negative distance as our metric so the max(- distance) = min(distance)
            if np.argmax(p) == np.argmax(y_test[i]):
                acc_tracker.update_state(1.0)
            else:
                acc_tracker.update_state(0.0)
            preds += [np.argmax(p)]
            targets += [np.argmax(y_test[i])]

        rec_tracker.update_state(_recall(preds,targets))
        pre_tracker.update_state(_precision(preds,targets))
        print(f'iteration {iteration+1}/{ITER}, accuracy:{acc_tracker.result()*100.0:.2f}, recall:{rec_tracker.result()*100.0:.2f}, precision:{pre_tracker.result()*100.0:.2f}')
    #print report in table
    tval = np.asarray(tval)
    tmean = tf.math.reduce_mean(tval)
    tstd = tf.math.reduce_std(tval)
    fpsval = np.asarray(fpsval)
    fpsmean = tf.math.reduce_mean(fpsval)
    fpsstd = tf.math.reduce_std(fpsval)
    #fps = 1.0 / tmean
    myTable = PrettyTable([" 1-NN Evaluation Report", ""])
    myTable.add_row(["Evaluation Data", data])
    myTable.add_row(["Top-1 Accuracy", f'{acc_tracker.result()*100.0:.2f}'])
    myTable.add_row(["Top-1 Recall", f'{rec_tracker.result()*100.0:.2f}'])
    myTable.add_row(["Top-1 Precision", f'{pre_tracker.result()*100.0:.2f}'])
    myTable.add_row(["Model Parameters", f'{count_params(loaded_encoder):.2f}M'])
    myTable.add_row(["Average Frame Rate", f'{fpsmean:.2f}FPS'])
    myTable.add_row(["Frame Rate std", f'{fpsstd:.2f}FPS'])
    myTable.add_row(["Average Inference Time", f'{tmean*1000:.1f}ms'])
    myTable.add_row(["Inference Time Std", f'{tstd*1000:.1f}ms'])
    print('\033[0;31m')
    print(myTable)
    print('\033[0m')
if __name__ == '__main__':

    if device == 'cpu':
        with tf.device('/cpu:0'):
            run_test(encoder_h5=encoder_file,data=config['test'], batch=config['batch_size'])
    elif device == 'gpu':
        with tf.device('/GPU:0'):
            run_test(encoder_h5=encoder_file,data=config['test'], batch=config['batch_size'])
    elif device == 'rpi':
        run_test(encoder_h5=encoder_file,data=config['test'], batch=config['batch_size'])
    