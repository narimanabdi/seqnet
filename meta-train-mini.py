###################################
#                                 #
# Meta Training for miniImageNet  #
#                                 #
###################################

from tensorflow import keras
import tensorflow as tf
import numpy as np
from tensorflow.keras.metrics import CategoricalAccuracy
from models.metrics import loss_mse
from models.makemodels import make_proto_model
import argparse
from gpu.gpu import set_gpu_memory_growth,GPU_info
from time import time
from data_loader import get_loader
from models.distances import Weighted_Euclidean_Distance
from models.stn import BilinearInterpolation,Localization

parser = argparse.ArgumentParser('Meta Training for miniImageNet')
parser.add_argument(
    '--epochs',type = int,required = True,help = 'Number of Training Epochs')

args = parser.parse_args()

#define parameters
backbone = 'densenet' #the name of TFE
lr = 1e-4
dim = 64 #input image dimension 84x84x3

def save_encoder(model_file):
    #save cascade encoder for KNN test
    encoder_h5 = 'model_files/best_' + backbone + '_mini' + '_encoder.h5'
    loaded_model = keras.models.load_model(
        model_file,
        custom_objects={
            'Weighted_Euclidean_Distance':Weighted_Euclidean_Distance,
            'BilinearInterpolation':BilinearInterpolation,
            'Localization':Localization},compile=False)
    enc = keras.Model(
        inputs=loaded_model.get_layer('encoder').input,
        outputs=loaded_model.get_layer('encoder').output)
    enc.save(encoder_h5)
    print(f'cascade encoder saved at {encoder_h5}')

def test(model,generator):
    acc = 0
    for z in generator:
        [Xs,Xq],y_true = z
        y_pred = model([Xs,Xq])
        acc += np.mean(keras.metrics.categorical_accuracy(y_true,y_pred))
    return acc/len(generator)

def train(model,generator,m,loss_tracker):
    pb = tf.keras.utils.Progbar(len(generator),verbose=1,stateful_metrics=['train loss','train acc'])
    for z in generator:
        [Xs,Xq],y_true = z
        with tf.GradientTape() as tape:
                #make prediction
                y_pred = model([Xs,Xq],training = True)
                loss =  loss_mse(y_true,y_pred)

        gradients = tape.gradient(loss,clf.trainable_variables)
        optimizer.apply_gradients(zip(gradients,clf.trainable_variables))
        loss_tracker.update_state(loss)
        m.update_state(y_true,y_pred)
        values=[('train loss',loss_tracker.result()),('train acc',m.result())]
        pb.add(1,values=values)
    m.reset_states()
    loss_tracker.reset_states()

loader = get_loader('mini')
train_datagen, test_datagen = loader.get_generator(batch=64,dim=dim)


if __name__ == "__main__":
    set_gpu_memory_growth()
    weights_h5 = 'model_files/best_' + backbone + '_mini' + '_weights.h5'
    model_h5 = 'model_files/best_' + backbone + '_mini' + '_whole.h5'
    

    best_test_acc,best_val_acc = [0,0]

    #make metric model
    clf = make_proto_model(backbone=backbone,input_shape=(dim,dim,3))
    clf.summary()
    train_metric = CategoricalAccuracy()
    optimizer = keras.optimizers.Adam(learning_rate=lr,epsilon=1.0e-8)
    train_loss_tracker = keras.metrics.Mean(name='train_loss')
    clf.compile(optimizer=optimizer,loss=loss_mse,metrics=train_metric)
   
    print('Starting meta training...')
    for step in range(5):
        print(f'Starting training stage {step + 1}')
        for epoch in range(args.epochs):
            print(f'epoch {epoch+1} / {args.epochs}')

            #meta train on seen data
            train(
                model=clf,generator=train_datagen,
                m=train_metric,loss_tracker=train_loss_tracker)
            #meta test on unseen data
            te_acc = test(model=clf,generator=test_datagen)
            #check best test accuracy
            if te_acc > best_test_acc:
                best_test_acc = te_acc
                clf.save_weights(weights_h5)#save whole metric model weights
                clf.save(model_h5)#save whole metric model
            print(f'best test accuracy: {best_test_acc:.4f}')
        
        #learning rate schedule
        clf.load_weights(weights_h5)
        optimizer.learning_rate = optimizer.learning_rate / 2

    print(f'Meta Training is ended')

    save_encoder(model_h5)  