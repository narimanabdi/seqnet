from tensorflow import keras
import tensorflow as tf
from tensorflow.keras.layers import TimeDistributed
import numpy as np
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.metrics import CategoricalAccuracy

import os
from tensorflow.keras.preprocessing.image import load_img,img_to_array
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt 
from datetime import datetime
from data_loader.gtsrb_template_loader import GTSRB_Template_DataGenerator
from data_loader.tt100k_template_loader import TT100K_Template_DataGenerator
from data_loader.persian_template_loader import PERSIAN_Template_DataGenerator
from models import encoders,protonet
import argparse

parser = argparse.ArgumentParser('This is test')
parser.add_argument('--backbone',required=True)
parser.add_argument('--weights',default=None)
parser.add_argument('--episode',default=800,required=True,type=int)
parser.add_argument('--p',default=25,type=int)

parser.add_argument('--outfile',help='name of output model file',required=True)
parser.add_argument('--epoch',type=int,help='name of output model file',required=True)

parser.add_argument('--freez',dest='freez',action='store_true',help='set augnmentation')
parser.set_defaults(freez=False)

parser.add_argument('--data',required=True)

args = parser.parse_args()

def test_step(model,generator):
    acc = 0
    loss = 0
    for z in generator:
        [Xs,Xq],y_true = z
        y_pred = model([Xs,Xq])
        acc += np.mean(keras.metrics.categorical_accuracy(y_true,y_pred))
        loss += loss_fun(y_true,y_pred)
    return loss/len(generator),acc/len(generator)

def loss_fun(y_true,y_pred):
    x = -tf.math.log(y_pred + 1e-12)
    return tf.reduce_mean(y_true*x)

def train_step(model,generator,m,loss_tracker):
    pb = tf.keras.utils.Progbar(len(generator),verbose=1,stateful_metrics=['train loss','train acc'])
    for z in generator:
        
        [Xs,Xq],y_true = z
        with tf.GradientTape() as tape:
                #make prediction
                y_pred = model([Xs,Xq],training = True)
                loss =  loss_fun(y_true,y_pred)
  
        #compute gradients of trainable parameters
        gradients = tape.gradient(loss,clf.trainable_variables)
        #update trainable parameters
        optimizer.apply_gradients(zip(gradients,clf.trainable_variables))
        #update mean loss value
        loss_tracker.update_state(loss)
        #update metric value e.g. accuracy
        m.update_state(y_true,y_pred)
        values=[('train loss',loss_tracker.result()),('train acc',m.result())]
        pb.add(1,values=values)
    m.reset_states()
    loss_tracker.reset_states()

def make_fs_model(backbone,weights):
    if backbone == 'resnet':
        encoder = encoders.create_resnet(input_shape=(64,64,3))

    if backbone == 'densenet':
        encoder = encoders.create_densenet(input_shape=(64,64,3),weights=weights)

    if backbone == 'minires':
        encoder = encoders.minires()
        encoder.summary()


    if backbone == '3conv_atn':
        encoder = encoders.create_3conv_atn(input_shape=(64,64,3),n_filters=args.p)

    
    model = protonet.create_model(input_shape = (64,64,3),encoder=encoder)
    model.load_weights('model_files/' + weights)
    return model



if __name__ == "__main__":

    clf = make_fs_model(backbone=args.backbone,weights=args.weights)

    if args.data == 'gtsrb':
        fine_datagen = GTSRB_Template_DataGenerator(N=21,Ks=1,Kq=1,num_episode=100,data_type='unseen')
    elif args.data == 'tt100k':
        fine_datagen = TT100K_Template_DataGenerator(n_way=36,k_shot=1,n_query=50,batch=128,data_type='all',target_size=(64,64))


    train_metric = CategoricalAccuracy()
    optimizer = keras.optimizers.Adam(learning_rate=1.0e-6,epsilon=1.0e-8)
    #optimizer = keras.optimizers.SGD(learning_rate=0.01,momentum=0.9)
    train_loss_tracker = keras.metrics.Mean(name='train_loss')

    clf.compile(optimizer=optimizer,loss=loss_fun,metrics=train_metric)

    file_name_val = args.outfile + "_val.h5"
    file_name_acc = args.outfile + "_acc.h5"
    print(f'model will be saved on {file_name_acc}')

    best_val_acc = 0.0
    best_val_loss = np.inf
    best_epoch = 1
    for epoch in range(args.epoch):
        
        print(f'epoch {epoch+1} / {args.epoch}')
        train_step(model=clf,generator=fine_datagen,m=train_metric,loss_tracker=train_loss_tracker)
        val_loss,acc = test_step(model=clf,generator=fine_datagen)
 
        print(f'val loss: {val_loss:.4f}, val accuracy: {acc:.4f}%')
        if val_loss <= best_val_loss:
            best_val_loss = val_loss
            clf.save_weights(file_name_val)
        if acc > best_val_acc:
            best_val_acc = acc
            best_epoch = epoch
            clf.save_weights(file_name_acc)
        print(f'best val loss: {best_val_loss:.4f}, best val accuracy: {best_val_acc:.4f}, best epoch = {best_epoch + 1}')