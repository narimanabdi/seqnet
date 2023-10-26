from tensorflow import keras
import tensorflow as tf
from tensorflow.keras.metrics import Mean, CategoricalAccuracy
from models.utils import make_senet_model, loss_mse
import argparse
from time import time
from data_loader import get_loader
from models.distances import Weighted_Euclidean_Distance
from models.stn import BilinearInterpolation,Localization
from tensorflow.keras.models import load_model
from models.senet import Senet


parser = argparse.ArgumentParser('SENet')
parser.add_argument('--backbone',type=str,default='densenet')
parser.add_argument('--test',type=str,default='belga2toplogoft')
parser.add_argument('--epochs',type=int,default=20)
parser.add_argument('--batch',type=int,default=128)
parser.add_argument('--dim',type=int,default=64)
parser.add_argument('--lr',type=float,default=1e-5)#3e-6

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

def meta_finetune(ep):
    #path to saved model
    model_ft_h5 = 'model_files/' + backbone + '_' + args.test + '.h5'
    encoder_ft_h5 = 'model_files/' + backbone + '_' + args.test + '_encoder.h5'
    senet_base = load_model(
        'model_files/best_models/densenet_belga2toplogo_whole.h5',
        custom_objects={
            'BilinearInterpolation':BilinearInterpolation,
            'Localization':Localization,
            'Senet':Senet,
            'Weighted_Euclidean_Distance':Weighted_Euclidean_Distance},compile=False)
    best_test_acc = 0.0

    optimizer_fn = keras.optimizers.Adam(learning_rate=lr,epsilon=1.0e-8)
    senet_base.compile(
        optimizer=optimizer_fn,loss_fn=loss_mse,
        metrics=CategoricalAccuracy(name = 'accuracy'))
    senet_base.get_layer('encoder').get_layer('model').trainable = False
    senet_base.summary()
    strat_time = time()
    train_datagen,test_datagen = make_data_generator(args.test)

    for epoch in range(ep):
        print(f'====epoch{epoch+1}/{ep}====')
        senet_base.fit(train_datagen)
        te_acc = senet_base.evaluate(test_datagen, verbose=0)
        if te_acc >= best_test_acc:
            best_test_acc = te_acc
            senet_base.save(model_ft_h5)
            senet_base.save_weights('best_weights.h5')
        print(f'test accuracy: {te_acc:.4f}')
        print(f'best test accuracy: {best_test_acc:.4f}')
    print('Meta-Training has just been ended')
    end_time = time() - strat_time
    print(f'trainig time: {end_time}')
    print(f'best test accuracy: {best_test_acc:.4f}')
    loaded_model = load_model(
        model_ft_h5,
        custom_objects={
            'Weighted_Euclidean_Distance':Weighted_Euclidean_Distance,
            'BilinearInterpolation':BilinearInterpolation,
            'Localization':Localization,
            'Senet':Senet},compile=False)
    enc = keras.Model(
        inputs=loaded_model.get_layer('encoder').input,
        outputs=loaded_model.get_layer('encoder').output)
    enc.save(encoder_ft_h5)
    print(f'sequential encoder saved at {encoder_ft_h5}')

if __name__ == "__main__":
    meta_finetune(args.epochs)

    