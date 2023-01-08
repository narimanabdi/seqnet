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
from gpu.gpu import set_gpu_memory_growth
from models import conv3b
import argparse

def GPU_info():
    gpus = tf.config.list_physical_devices('GPU')
    
    if gpus:
        try:
            for gpu in gpus:
                print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
                print("Device Name: ", tf.config.experimental.get_device_details(gpu)['device_name'])
                print("Compute Capability: ", tf.config.experimental.get_device_details(gpu)['compute_capability'])
                print("Device Policy: ",tf.config.experimental.get_device_policy())
                print("Memory Growth: ",tf.config.experimental.get_memory_growth(gpu))
        except RuntimeError as e:
            print(e)
    else:
        print('GPU not found')

def gpu_mem_grow():
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)

parser = argparse.ArgumentParser('This is test')
parser.add_argument('--backbone',required=True)
parser.add_argument('--epoch',default=50,help='enter epoch numbers',type=int)
parser.add_argument('--batch',default=32,help='batch size of data',type=int)
parser.add_argument('--augmentation',dest='aug',action='store_true',help='set augnmentation')
parser.add_argument('--no-augmentation',dest='aug',action='store_false',help='reset augnmentation')
parser.set_defaults(aug=True)
parser.add_argument('--data',default='tiny',choices=['tiny','mini','tiered','minigt'], help='path of data')
parser.add_argument('--targetsize',default=224,type=int)
parser.add_argument('--lr',default=1.0e-2,type=float)


parser.add_argument('--outfile',help='name of output model file',required=True)


args = parser.parse_args()

def test_step(generator,model,loss_fun):
    acc = 0
    loss = 0
    for i,z in enumerate(generator):
        if i >= len(generator):
            break
        X,y_true = z
        y_pred = model(X)
        acc += np.mean(keras.metrics.categorical_accuracy(y_true,y_pred))
        loss += loss_fun(y_true,y_pred)
    return loss/len(generator),acc/len(generator)

def train_step(model,X,y_true,loss_tracker,metrics,loss_fun):
    with tf.GradientTape() as tape:
            #make prediction
            y_pred = model(X,training = True)
            loss =  loss_fun(y_true,y_pred)

    #compute gradients of trainable parameters
    gradients = tape.gradient(loss,model.trainable_variables)
    #update trainable parameters
    optimizer.apply_gradients(zip(gradients,model.trainable_variables))
    #update mean loss value
    loss_tracker.update_state(loss)
    #update metric value e.g. accuracy
    metrics.update_state(y_true,y_pred)
    return train_loss_tracker.result(),metrics.result()

def make_vanilla_model(backbone,n_class,targetsize):
    if backbone == 'resnet34':
        encoder = encoders_prefinal.create_resnet34(input_shape=(targetsize,targetsize,3))

    if backbone == 'resnet18':
        encoder = encoders_prefinal.create_resnet18(input_shape=(targetsize,targetsize,3))

    
    if backbone == 'resnet12':
        encoder = encoders_prefinal.create_resnet12(input_shape=(targetsize,targetsize,3))

    if backbone == 'resnet10':
        encoder = encoders_prefinal.create_resnet10(input_shape=(targetsize,targetsize,3))
        
    if backbone == 'resnet8':
        encoder = encoders_prefinal.create_resnet8(input_shape=(targetsize,targetsize,3))
        
    
    if backbone == 'densenet121':
        encoder = encoders_prefinal.create_densenet121(input_shape=(targetsize,targetsize,3))

    if backbone == '3conv_atn':
        encoder = encoders_prefinal.create_3conv_atn(input_shape=(targetsize,targetsize,3),n_filters=args.p)
    
    if backbone == 'conv3b':
        encoder = conv3b.create_conv3b(input_shape=(64,64,3))

    if backbone == 'conv64':
        encoder = encoders_prefinal.create_conv64f(input_shape=(targetsize,targetsize,3))
    
    if backbone == 'conv256':
        encoder = encoders_prefinal.create_conv256(input_shape=(targetsize,targetsize,3))
    if backbone == 'conv8':
        encoder = encoders_prefinal.create_conv8b(input_shape=(targetsize,targetsize,3))
    if backbone == 'conv16':
        encoder = encoders_prefinal.create_conv16b(input_shape=(targetsize,targetsize,3))
    if backbone == 'dires':
        encoder = encoders_prefinal.create_dires(input_shape=(targetsize,targetsize,3))

    if backbone == 'dires2':
        encoder = encoders_prefinal.create_dires2(input_shape=(targetsize,targetsize,3))
    
    if backbone == 'myconv':
        encoder = encoders_prefinal.create_myconv(input_shape=(targetsize,targetsize,3))

    if backbone == 'myconv2':
        encoder = encoders_prefinal.create_myconv2(input_shape=(targetsize,targetsize,3))
    
    if backbone == 'conv4atn':
        encoder = encoders_prefinal.create_4conv_atn(input_shape=(targetsize,targetsize,3))

    if backbone == 'convvpe':
        encoder = encoders_prefinal.create_conv_vpe(input_shape=(targetsize,targetsize,3))
    
    encoder.summary()

    inp = keras.layers.Input([targetsize,targetsize,3])
    x = encoder(inp)
    #x = keras.layers.GlobalAveragePooling2D()(x)
    #x = keras.layers.Dense(2000,activation='relu')(x)
    x = keras.layers.Dense(n_class,activation='softmax')(x)
    
    return keras.Model(inp,x),encoder



if __name__ == "__main__":
    set_gpu_memory_growth()
    #gpu_mem_grow()
    GPU_info()
    global n_cls, datapath
    if args.data == 'tiered':
        datapath = 'datasets/tiered_Imagenet/train'
        if args.aug:
            datagen = ImageDataGenerator(validation_split=0.02,rotation_range = 20,shear_range=0.2,height_shift_range=0.1,width_shift_range=0.1,horizontal_flip=True)
        else:
            datagen = ImageDataGenerator(validation_split=0.02)

        train_gen = datagen.flow_from_directory(datapath,batch_size=args.batch,class_mode='categorical',target_size=(args.targetsize,args.targetsize),subset='training')
        val_gen = datagen.flow_from_directory(datapath,batch_size=256,class_mode='categorical',target_size=(args.targetsize,args.targetsize),subset='validation')
        n_cls = 351
    
    if args.data == 'tiny':
        datapath = 'datasets/tinyImageNet'
        if args.aug:
            datagen = ImageDataGenerator(validation_split=0.02,rotation_range = 20,shear_range=0.2,height_shift_range=0.1,width_shift_range=0.1,horizontal_flip=True)
        else:
            datagen = ImageDataGenerator(validation_split=0.02)

        train_gen = datagen.flow_from_directory(datapath,batch_size=args.batch,class_mode='categorical',target_size=(args.targetsize,args.targetsize),subset='training')
        val_gen = datagen.flow_from_directory(datapath,batch_size=256,class_mode='categorical',target_size=(args.targetsize,args.targetsize),subset='validation')
        n_cls = 200
    if args.data == 'mini':
        datapath = 'datasets/miniImageNet/all'
        if args.aug:
            datagen = ImageDataGenerator(validation_split=0.02,rotation_range = 20,shear_range=0.2,height_shift_range=0.1,width_shift_range=0.1,horizontal_flip=True)
        else:
            datagen = ImageDataGenerator(validation_split=0.02)

        train_gen = datagen.flow_from_directory(datapath,batch_size=args.batch,class_mode='categorical',target_size=(args.targetsize,args.targetsize),subset='training')
        val_gen = datagen.flow_from_directory(datapath,batch_size=256,class_mode='categorical',target_size=(args.targetsize,args.targetsize),subset='validation')
        n_cls = 100

    if args.data == 'minigt':
        datapath = 'datasets/minigt'
        if args.aug:
            datagen = ImageDataGenerator(validation_split=0.02,rotation_range = 20,shear_range=0.2,height_shift_range=0.1,width_shift_range=0.1,horizontal_flip=True)
        else:
            datagen = ImageDataGenerator(validation_split=0.02)

        train_gen = datagen.flow_from_directory(datapath,batch_size=args.batch,class_mode='categorical',target_size=(args.targetsize,args.targetsize),subset='training')
        val_gen = datagen.flow_from_directory(datapath,batch_size=256,class_mode='categorical',target_size=(args.targetsize,args.targetsize),subset='validation')
        n_cls = 143
 
    clf_encoder,encoder = make_vanilla_model(backbone=args.backbone,n_class=n_cls,targetsize=args.targetsize)

    metric = CategoricalAccuracy()
    train_loss_tracker = keras.metrics.Mean(name='loss')
    optimizer = keras.optimizers.SGD(learning_rate=args.lr,momentum=0.9)
    #optimizer = keras.optimizers.SGD(learning_rate=0.1,momentum=0.5)
    #optimizer = keras.optimizers.Adam(learning_rate=0.001,epsilon=1.0e-8)
    loss_fun = keras.losses.CategoricalCrossentropy()
    clf_encoder.compile(optimizer=optimizer,loss=loss_fun,metrics=metric)


    file_name_val = 'model_files/' + args.outfile + "_val.h5"
    file_name_acc = 'model_files/' + args.outfile + "_acc.h5"
    file_name_last = 'model_files/' + args.outfile + "_last.h5"
    print(f'model will be saved on {file_name_acc}')

    best_val_loss = np.inf
    best_val_acc = 0.0
    best_epoch = 1
    #old_val_acc = 0.0
    #plateu = 0
    for epoch in range(args.epoch):
        print(f'epoch {epoch+1} / {args.epoch}')
        pb = tf.keras.utils.Progbar(len(train_gen),verbose=1,stateful_metrics=['train loss','train acc'])
        for i,z in enumerate(train_gen):
            if i >= len(train_gen):
                break
            x,y = z
            loss,m = train_step(model=clf_encoder,X=x,y_true=y,loss_tracker=train_loss_tracker,metrics=metric,loss_fun=loss_fun)
            values=[('train loss',loss),('train acc',m)]
            pb.add(1,values=values)
        metric.reset_states()
        val_loss,acc = test_step(generator=val_gen,model=clf_encoder,loss_fun=loss_fun)
        
        if val_loss <= best_val_loss:
            best_val_loss = val_loss
            encoder.save_weights(file_name_val)
        if acc >= best_val_acc:
            best_val_acc = acc
            encoder.save_weights(file_name_acc)
            print('encoder saved')
            best_epoch = epoch
        
        #if acc >= old_val_acc:
           # old_val_acc = acc
            #plateu = 0
        #if acc < old_val_acc:
            #plateu += 1
        #if plateu > 5:
            #optimizer.learning_rate = 0.5 * optimizer.learning_rate
            #plateu = 0

        #if optimizer.learning_rate > 3.0e-6:
            #optimizer.learning_rate = (0.01)/(0.993*(epoch + 1))
        #else:
            #optimizer.learning_rate = 3.0e-6
        if epoch == 90:
            optimizer.learning_rate = optimizer.learning_rate * 0.1
        print(f'val acc: {acc:.4f}, best val loss: {best_val_loss:.4f}, best val accuracy: {best_val_acc:.4f}, best epoch = {best_epoch + 1},learning rate:{optimizer.learning_rate.numpy():.4f} ')
    encoder.save_weights(file_name_last)
    print('final encoder saved')
    print('Training has beeen finished')