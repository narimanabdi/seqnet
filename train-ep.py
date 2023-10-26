from tensorflow import keras
import tensorflow as tf
from tensorflow import keras
from keras.metrics import Mean, CategoricalAccuracy
from models.utils import make_senet_model, loss_ce
import argparse
from time import time
from data_loader import get_loader
from models.distances import Weighted_Euclidean_Distance
from models.stn import BilinearInterpolation,Localization
from tensorflow.keras.models import load_model
from models.senet import Senet


parser = argparse.ArgumentParser('SENet')
parser.add_argument('--backbone',type=str,default='densenet')
parser.add_argument('--test',type=str,default='gtsrb2tt100k')
parser.add_argument('--epochs',type=int,default=50)
parser.add_argument('--batch',type=int,default=128)
parser.add_argument('--dim',type=int,default=64)
parser.add_argument('--lr',type=float,default=1e-4)

args = parser.parse_args()

#hyperparameters
backbone = args.backbone
dim = args.dim
batch = args.batch
epochs = args.epochs
lr = args.lr
#metric tracker
train_acc_tracker = Mean('train_accuracy')
train_loss_tracker = Mean('train_loss')
test_acc_tracker = Mean('train_accuracy')

def make_data_generator(test_data):

    loader = get_loader(test_data) 
    if test_data== 'belga2flick' or \
        test_data == 'belga2toplogo' or test_data == 'gtsrb':
        train_datagen, val_datagen,test_datagen = loader.get_generator(
            batch=batch,dim=dim)
    else:
        train_datagen, test_datagen = loader.get_generator(
            batch=batch,dim=dim)

    return train_datagen ,test_datagen

def meta_train(ep):
    #path to saved model
    model_h5 = 'model_files/' + backbone + '_' + args.test + '.h5'
    encoder_h5 = 'model_files/' + backbone + '_' + args.test + '_encoder.h5'

    best_test_acc = 0.0

    senet = make_senet_model(backbone=backbone,input_shape=(dim,dim,3),
                             truncated_layer="conv4_block14_concat")
    optimizer_fn = keras.optimizers.Adam(learning_rate=lr,epsilon=1.0e-8)
    senet.compile(
        optimizer=optimizer_fn,loss_fn=loss_ce,
        metrics=CategoricalAccuracy(name = 'accuracy'))
    
    strat_time = time()
    train_datagen,test_datagen = make_data_generator('mini')
    for step in range(1):
        print(f'=====step {step+1}=====')
        for epoch in range(ep):
            print(f'====epoch{epoch+1}/{ep}====')
            senet.fit(train_datagen)
            te_acc = senet.evaluate(test_datagen, verbose=0)
            if te_acc >= best_test_acc:
                best_test_acc = te_acc
                senet.save(model_h5)
                senet.save_weights('best_weights.h5')
            print(f'test accuracy: {te_acc:.4f}')
            print(f'best test accuracy: {best_test_acc:.4f}')
        senet.load_weights('best_weights.h5')
        optimizer_fn.learning_rate = optimizer_fn.learning_rate / 2   
   
    print('Meta-Training has just been ended')
    end_time = time() - strat_time
    print(f'trainig time: {end_time}')
    print(f'best test accuracy: {best_test_acc:.4f}')
    loaded_model = load_model(
        model_h5,
        custom_objects={
            'Weighted_Euclidean_Distance':Weighted_Euclidean_Distance,
            'BilinearInterpolation':BilinearInterpolation,
            'Localization':Localization,
            'Senet':Senet},compile=False)
    enc = keras.Model(
        inputs=loaded_model.get_layer('encoder').input,
        outputs=loaded_model.get_layer('encoder').output)
    enc.save(encoder_h5)
    print(f'sequential encoder saved at {encoder_h5}')

if __name__ == "__main__":
    meta_train(args.epochs)

    