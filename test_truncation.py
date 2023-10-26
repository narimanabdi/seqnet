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
import numpy as np


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
loader = get_loader('gtsrb2tt100k') 
train_datagen,test_datagen = loader.get_generator(batch=batch,dim=dim)
truncated_layer_list = ['conv4_block1_concat','conv4_block2_concat']
def meta_train(ep):
    
    for truncated_id in truncated_layer_list:
        senet = make_senet_model(backbone=backbone,input_shape=(dim,dim,3),truncated_layer=truncated_id)
        optimizer_fn = keras.optimizers.Adam(learning_rate=lr,epsilon=1.0e-8)
        senet.compile(
            optimizer=optimizer_fn,loss_fn=loss_mse,
            metrics=CategoricalAccuracy(name = 'accuracy'))
        best_test_acc = 0.0
        final_acc = []
        for epoch in range(ep):
            print(f'====epoch{epoch+1}/{ep}====')
            senet.fit(train_datagen)
            te_acc = senet.evaluate(test_datagen, verbose=0)
            if te_acc > best_test_acc:
                best_test_acc = te_acc
                #senet.save(model_h5)
                #senet.save_weights('best_weights.h5')
            #print(f'test accuracy: {te_acc:.4f}')
            #print(f'best test accuracy: {best_test_acc:.4f}')
            final_acc.append(best_test_acc)
        #senet.load_weights('best_weights.h5')
        #optimizer_fn.learning_rate = optimizer_fn.learning_rate / 2   
        final_acc = np.asarray(final_acc)
        print(f'mean accuracy of {truncated_id} is {final_acc.mean():.4f}')
        print(f'best accuracy of {truncated_id} is {best_test_acc:.4f}')
    
if __name__ == "__main__":
    meta_train(args.epochs)

    