from tensorflow import keras
import tensorflow as tf
import numpy as np
from tensorflow.keras.metrics import CategoricalAccuracy
import os
from data_loader.gtsrb_template_loader import GTSRB_Template_DataGenerator
from data_loader.tt100k_template_loader import TT100K_Template_DataGenerator
from data_loader.mini_loader2 import   MINI_DataGenerator
#from data_loader.persian_template_loader import PERSIAN_Template_DataGenerator
from models import resnet,densenet,conv3b,mobilenet,vggnet
import argparse
from gpu.gpu import set_gpu_memory_growth,GPU_info
from time import time

parser = argparse.ArgumentParser('This is test')
parser.add_argument('--backbone',required=True)
parser.add_argument('--lr',default=2.0e-4,type=float)
parser.add_argument('--outfile',help='name of output model file',required=True)
parser.add_argument('--epoch',type=int,help='name of output model file',required=True)
parser.add_argument('--data',required=True)

args = parser.parse_args()

def test_step(model,generator):
    acc = 0
    loss = 0
    #pb = tf.keras.utils.Progbar(len(generator),verbose=1)
    for z in generator:
        [Xs,Xq],y_true = z
        y_pred = model([Xs,Xq])
        acc += np.mean(keras.metrics.categorical_accuracy(y_true,y_pred))
        loss += loss_fun(y_true,y_pred)
        #pb.add(1)
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

def make_fs_model(backbone):
    if backbone == 'resnet':
        return resnet.create_model(input_shape = (64,64,3))
    if backbone == 'densenet':
        return densenet.create_model(input_shape = (64,64,3))
    if backbone == 'conv3b':
        return conv3b.create_model(input_shape = (64,64,3))
    if backbone == 'vgg':
        return vggnet.create_model(input_shape = (64,64,3))
    if backbone == 'mobilenet':
        return mobilenet.create_model(input_shape = (64,64,3))

if __name__ == "__main__":
    file_name_val = 'model_files/' + args.outfile + "_val.h5"
    file_name_acc = 'model_files/' + args.outfile + "_acc.h5"
    file_name_last = 'model_files/' + args.outfile + "_last.h5"

    best_val_acc = 0.0
    best_val_loss = np.inf
    best_epoch = 1
    val_acc = 0

    #set_gpu_memory_growth()
    GPU_info()

    clf = make_fs_model(backbone=args.backbone)

    if args.data == 'gtsrb':
        train_datagen = GTSRB_Template_DataGenerator(n_way=22,k_shot=1,n_query=50,batch=128,data_type='seen',target_size=(64,64),augmentation=True)
        val_datagen = GTSRB_Template_DataGenerator(n_way=21,k_shot=1,n_query=50,batch=128,data_type='unseen',target_size=(64,64))
    if args.data == 'tt100k':
        train_datagen = GTSRB_Template_DataGenerator(n_way=43,k_shot=1,n_query=50,batch=128,data_type='all',target_size=(64,64),augmentation=True)
        val_datagen = TT100K_Template_DataGenerator(n_way=36,k_shot=1,n_query=50,batch=128,data_type='all',target_size=(64,64))
    if args.data == 'mini':
        train_datagen = MINI_DataGenerator(n_way=5,k_shot=5,n_query=10,num_episode=1000,data_type='seen',target_size=(64,64),augmentation=True)
        val_datagen = MINI_DataGenerator(n_way=5,k_shot=5,n_query=10,num_episode=400,data_type='unseen',target_size=(64,64))

    
    train_metric = CategoricalAccuracy()
    optimizer = keras.optimizers.Adam(learning_rate=args.lr,epsilon=1.0e-8)
    #optimizer = keras.optimizers.SGD(learning_rate=args.lr,momentum=0.9)
    train_loss_tracker = keras.metrics.Mean(name='train_loss')

    clf.compile(optimizer=optimizer,loss=loss_fun,metrics=train_metric)
    
    clf.summary()
    print('Starting training')
    strat_time = time()
    for epoch in range(args.epoch):

        
        print(f'epoch {epoch+1} / {args.epoch}')
        print(f'learning rate: {optimizer.learning_rate.numpy()}')
        train_step(model=clf,generator=train_datagen,m=train_metric,loss_tracker=train_loss_tracker)
        val_loss,acc = test_step(model=clf,generator=val_datagen)
        val_acc += acc
        print(f'val loss: {val_loss:.4f}, val accuracy: {acc:.4f}%')
        if val_loss <= best_val_loss:
            best_val_loss = val_loss
            clf.save_weights(file_name_val)
        if np.round(acc,4) >= best_val_acc:
            best_val_acc = np.round(acc,4)
            best_epoch = epoch
            clf.save_weights(file_name_acc)
        print(f'best val loss: {best_val_loss:.4f}, best val accuracy: {best_val_acc:.4f},mean val accuracy: { val_acc / (epoch + 1):.4f} ,best epoch = {best_epoch + 1}')


        
    end_time = time() - strat_time
    print(f'trainig time: {end_time}')
    home = os.getcwd()
    path = os.path.join(home,'reports')
    file_name = args.backbone + '_metatrain_report.txt'
    file_path = os.path.join(path,file_name)
    with open(file_path,'a') as f:
        print('+--------------------+',file=f)
        print('|Meta Training Report|',file=f)
        print('+--------------------+',file=f)
        print(f'| Pretrained on | ImageNet|',file=f)
        print('+--------------------+',file=f)
        print(f'| Backbone | {args.backbone}|',file=f)
        print('+--------------------+',file=f)
        print(f'|Accuracy | {best_val_acc:.4f}|',file=f)
        print('+--------------------+',file=f)
    clf.save_weights(file_name_last)