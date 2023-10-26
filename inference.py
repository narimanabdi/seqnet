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

parser = argparse.ArgumentParser('Nearest Neighbor Test')
parser.add_argument('--data',type = str,default='gtsrb2tt100k',help='Test Data')
parser.add_argument('--device',type = str,default='cpu',choices=['cpu','gpu','rpi'])
parser.add_argument('--dist',default='l2',choices=['l2','cosine'])
parser.add_argument(
    '--backbone',required=True,help='Backbone Network',
    choices=['densenet','mobilenet','resnet','densenetmini'])
args = parser.parse_args() 

if args.dist == 'l2':
    dist_fn = Euclidean_Distance()
elif args.dist == 'cosine':
    dist_fn = Cosine_Distance()
#metric tracker
acc_tracker = Mean(name='Nearest_Neighbor_Accuracy')
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
    pb = tf.keras.utils.Progbar(len(test_generator),verbose=1)
    tval = []
    fpsval = []
    for d,y_test in test_generator:
        for i,x in enumerate(d[1]):
            s = time()
            p = nn(loaded_encoder,x,Zs)
            tval.append(time() - s)
            fpsval.append(1.0 / tval[-1])
            if np.argmax(p) == np.argmax(y_test[i]):
                acc_tracker.update_state(1.0)
            else:
                acc_tracker.update_state(0.0)
        pb.add(1)
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
    myTable.add_row(["Model Parameters", f'{count_params(loaded_encoder):.2f}M'])
    myTable.add_row(["Average Frame Rate", f'{fpsmean:.2f}FPS'])
    myTable.add_row(["Frame Rate std", f'{fpsstd:.2f}FPS'])
    myTable.add_row(["Average Inference Time", f'{tmean*1000:.1f}ms'])
    myTable.add_row(["Inference Time Std", f'{tstd*1000:.1f}ms'])
    print('\033[0;31m')
    print(myTable)
    print('\033[0m')

if __name__ == '__main__':
    #encoder_file = 'model_files/best_encoders/' + args.backbone + '_' + args.data + '_encoder.h5'
    encoder_file = 'model_files/best_encoders/densenet_tt100kft_encoder.h5'
    if args.device == 'cpu':
        with tf.device('/cpu:0'):
            run_test(encoder_h5=encoder_file,data=args.data, batch=128)
    elif args.device == 'gpu':
        with tf.device('/GPU:0'):
            run_test(encoder_h5=encoder_file,data=args.data, batch=256)
    elif args.device == 'rpi':
        run_test(encoder_h5=encoder_file,data=args.data, batch=16)
    