###############################
#                             #
# Nearest Neighbor Evaluation #
#                             #
###############################

from time import time
from data_loader import get_loader
from tensorflow.keras.models import load_model
from models.stn import BilinearInterpolation,Localization
from sklearn.neighbors import KNeighborsClassifier
from tensorflow.keras.utils import to_categorical
import numpy as np
import tensorflow as tf
from tensorflow.keras.metrics import Mean
from prettytable import PrettyTable
from models.utils import count_params
import argparse
from models.distances import Weighted_Euclidean_Distance, Euclidean_Distance

parser = argparse.ArgumentParser('Nearest Neighbor Test')
parser.add_argument('--data',type = str,default='gtsrb2tt100k',help='Test Data')
parser.add_argument('--device',type = str,choices=['normal','rpi'])
parser.add_argument('--metric',default='l2',choices=['l2','cosine'])
parser.add_argument(
    '--backbone',required=True,help='Backbone Network',
    choices=['densenet','mobilenet','resnet','densenetmini'])
args = parser.parse_args() 

#metric tracker
acc_tracker = Mean(name='Nearest_Neighbor_Accuracy')
time_tracker = Mean(name='Time')
@tf.function
def nn(model,inp,ztemplates):
    z = model(tf.expand_dims(inp,axis=0))
    return Euclidean_Distance()([ztemplates,z])

def fps_benchmark(encoder_h5,data,batch):
    loaded_encoder = load_model(
        encoder_h5,
        custom_objects={
            'BilinearInterpolation':BilinearInterpolation,
            'Localization':Localization},compile=False)

    loader = get_loader(data) 
    test_generator = loader.get_test_generator(batch=batch,dim=64,shuffle=False)
    t = iter(test_generator)
    [Xs,Xq],y = next(t)
    del y

    Zs = loaded_encoder(Xs)
    n_cls = len(Xs)
    nn = KNeighborsClassifier(n_neighbors=1,n_jobs=-1,metric = args.metric)
    y_train = to_categorical(np.arange(n_cls),num_classes=n_cls)
    nn.fit(Zs,y_train);

    print('\033[3;34mStart FPS Benchmark\033[0m')
    fps = 0
    for _,x in enumerate(Xq):
        start = time()
        zq = loaded_encoder(tf.expand_dims(x,axis=0))
        y_pred = nn.predict(zq)
        time_tracker.update_state(time() - start)
    time_per_image = time_tracker.result()
    fps += 1/time_per_image
    return fps

def run_test(encoder_h5,data,batch):
    #fps = fps_benchmark(encoder_h5=encoder_h5, data=data, batch=2*batch)
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
    n_cls = len(Xs)
    #nn = KNeighborsClassifier(n_neighbors=1,n_jobs=-1,metric = args.metric)
    #y_train = to_categorical(np.arange(n_cls),num_classes=n_cls)
    #nn.fit(Zs,y_train);
    del Xs
    print('\033[0;32mStart Nearest Neighbor Evaluation\033[0m')
    pb = tf.keras.utils.Progbar(len(test_generator),verbose=1)
    tval = []
    for d,y_test in test_generator:
        #[Xs,Xq],y_test = d
        #Zq = loaded_encoder(Xq)
        #acc_tracker.update_state(nn.score(Zq,y_test)) 
        for i,x in enumerate(d[1]):
            s = time()
            #Zq[i] = original_encoder(tf.expand_dims(x,axis=0))
            #acc_tracker.update_state(nn.score(tf.expand_dims(Zq[i],axis=0),tf.expand_dims(y_test[i],axis=0)))
            #p = nn.predict(tf.expand_dims(Zq[i],axis=0))
            p = nn(loaded_encoder,x,Zs)
            tval.append(time() - s)
            if np.argmax(p) == np.argmax(y_test[i]):
                acc_tracker.update_state(1.0)
            else:
                acc_tracker.update_state(0.0)
        pb.add(1)
    #print report in table
    tval = np.asarray(tval)
    tmean = tf.math.reduce_mean(tval)
    tstd = tf.math.reduce_std(tval)
    fps = 1.0 / tmean
    myTable = PrettyTable([" 1-NN Evaluation Report", ""])
    myTable.add_row(["Evaluation Data", data])
    myTable.add_row(["Top-1 Accuracy", f'{acc_tracker.result()*100.0:.2f}'])
    myTable.add_row(["Model Parameters", f'{count_params(loaded_encoder):.2f}M'])
    myTable.add_row(["Frame Rate", f'{fps:.2f}FPS'])
    myTable.add_row(["Average Inference Time", f'{tmean*1000:.1f}ms'])
    myTable.add_row(["Inference Time Std", f'{tstd*1000:.1f}ms'])
    print('\033[0;31m')
    print(myTable)
    print('\033[0m')

if __name__ == '__main__':
    encoder_file = 'model_files/best_encoders/' + args.backbone + '_' + args.data + '_encoder.h5'
    if args.device == 'normal':
        run_test(encoder_h5=encoder_file,data=args.data, batch=128)
    elif args.device == 'rpi':
        run_test(encoder_h5=encoder_file,data=args.data, batch=32)
    