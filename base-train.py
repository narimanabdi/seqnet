from tensorflow import keras
import tensorflow as tf
import numpy as np
from tensorflow.keras.metrics import CategoricalAccuracy
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import argparse
from models.makemodels import make_base_model

parser = argparse.ArgumentParser('This is test')
parser.add_argument('--epochs',type = int,default=100)

args = parser.parse_args()

dim = 64
lr = 1e-1
epochs = args.epochs
batch = 128

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

def get_n_classes(datapath):
    return len(os.listdir(datapath))

if __name__ == "__main__":
    datapath = 'datasets/mini/all'
    tr_datagen = ImageDataGenerator(validation_split=0.02,rotation_range = 20,shear_range=0.2,height_shift_range=0.1,width_shift_range=0.1,horizontal_flip=True)

    train_gen = tr_datagen.flow_from_directory(datapath,batch_size=batch,class_mode='categorical',target_size=(dim,dim),subset='training')
    val_gen = tr_datagen.flow_from_directory(datapath,batch_size=batch,class_mode='categorical',target_size=(dim,dim),subset='validation')
    n_cls = get_n_classes(datapath)
    print(f'\033[0;31mnumber of classes {n_cls}\033[0m')
 
    clf_encoder,encoder = make_base_model(backbone='densenet',n_class=n_cls,dim=dim)
    clf_encoder.summary()

    metric = CategoricalAccuracy()
    train_loss_tracker = keras.metrics.Mean(name='loss')
    optimizer = keras.optimizers.SGD(learning_rate=lr,momentum=0.9)
    loss_fun = keras.losses.CategoricalCrossentropy()
    clf_encoder.compile(optimizer=optimizer,loss=loss_fun,metrics=metric)


    file_best = 'model_files/best_densent_encoder_mini_pretrained.h5'
    print(f'model will be saved on {file_best}')

    best_val_loss = np.inf
    best_val_acc = 0.0

    for epoch in range(epochs):
        print(f'epoch {epoch+1} / {epochs}')
        clf_encoder.fit(train_gen,epochs=1)
        val_loss,acc = test_step(generator=val_gen,model=clf_encoder,loss_fun=loss_fun)

        if acc > best_val_acc:
            best_val_acc = acc
        encoder.save(file_best)
        if epoch == 90:
            optimizer.learning_rate = optimizer.learning_rate * 0.1
        
        print(f'val acc: {acc:.4f}, best val accuracy: {best_val_acc:.4f}')
    print('Training has beeen finished')