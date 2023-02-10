from tensorflow import keras
import tensorflow as tf
from tensorflow.keras.metrics import Mean, CategoricalAccuracy
from models.metrics import loss_mse,accuracy
from models.makemodels import make_proto_model
import argparse
from gpu.gpu import set_gpu_memory_growth
from time import time
from utils import load_config
from data_loader import get_loader
from models.distances import Weighted_Euclidean_Distance
from models.stn import BilinearInterpolation,Localization

parser = argparse.ArgumentParser('SENet')
#parser.add_argument('--cfg',type = str,required = True,help = 'config file')
parser.add_argument('--backbone',type=str,default='densenet')
parser.add_argument('--test',type=str,default='gtsrb2tt100k')
parser.add_argument('--epochs',type=int,default=50)
parser.add_argument('--batch',type=int,default=128)
parser.add_argument('--dim',type=int,default=64)
parser.add_argument('--lr',type=float,default=1e-4)

args = parser.parse_args()

#load config file and import training setting
#cfg = load_config(args.cfg)
backbone = args.backbone#cfg['backbone']
dim = args.dim#cfg['dim']
batch = args.batch#cfg['batch']
epochs = args.epochs#cfg['epochs']
lr = args.lr#cfg['lr']
#test_mode = args.test
#learning_rates = np.array([1.0e-4,5.0e-5,2.0e-5,1.0e-5])
#backbone = args.backbone #cfg['backbone']
#dim = args.dim#cfg['dim']

#batch = args.batch#cfg['batch']
#epochs = args.epochs

#def accuracy fuunction
train_acc_tracker = Mean('train_accuracy')
train_loss_tracker = Mean('train_loss')
test_acc_tracker = Mean('train_accuracy')



#test function
def test(model,generator):
    acc_mean = tf.constant([0.0],dtype='float32')
    #test_acc_tracker = Mean('test_accuracy')
    for z in generator:
        [Xs,Xq],y_true = z
        y_pred = model([Xs,Xq])
        #acc += accuracy(y_true,y_pred)
        test_acc_tracker.update_state(accuracy(y_true,y_pred))
    acc_mean = test_acc_tracker.result()
    test_acc_tracker.reset_state()
    return acc_mean
    #return test_acc_tracker.result()

#train function
@tf.function
def train_step(model,data,optimizer):
    [Xs,Xq],y_true = data
    with tf.GradientTape() as tape:
        #make prediction
        y_pred = model([Xs,Xq],training = True)
        loss =  loss_mse(y_true,y_pred)
    gradients = tape.gradient(loss,model.trainable_variables)
    optimizer.apply_gradients(zip(gradients,model.trainable_variables))
    return loss,accuracy(y_true,y_pred)

def train(model,generator,optimizer):
    pb = tf.keras.utils.Progbar(
        len(generator),verbose=1,stateful_metrics=['train loss','train acc'])
    for data in generator:
        loss,acc = train_step(model,data=data,optimizer=optimizer)
        #update mean loss value
        train_loss_tracker.update_state(loss)
        #update mean acc value
        train_acc_tracker.update_state(acc)
        values=[('train loss',train_loss_tracker.result()),('train acc',train_acc_tracker.result())]
        pb.add(1,values=values)
    train_loss_tracker.reset_states()
    train_acc_tracker.reset_states()

def make_data_generator(test_mode):
    #validation = False
    loader = get_loader(test_mode) 
    if test_mode== 'belga2flick' or \
        test_mode == 'belga2toplogo' or test_mode == 'gtsrb':
        train_datagen, val_datagen,test_datagen = loader.get_generator(
            batch=batch,dim=dim)
        #return train_datagen, val_datagen,test_datagen
        #validation = False
    else:
        train_datagen, test_datagen = loader.get_generator(
            batch=batch,dim=dim)

    return train_datagen ,test_datagen


if __name__ == "__main__":
    #weights_h5 = 'model_files/' + backbone + '_' + args.test + '_weights.h5'
    model_h5 = 'model_files/' + backbone + '_' + args.test + '_whole.h5'
    encoder_h5 = 'model_files/' + backbone + '_' + args.test + '_encoder.h5'

    best_test_acc = 0.0
    best_val_acc = 0.0

    senet = make_proto_model(backbone=backbone,input_shape=(dim,dim,3))
    senet.summary()
    #train_metric = CategoricalAccuracy()
    optimizer_fn = keras.optimizers.Adam(learning_rate=lr,epsilon=1.0e-8)
    #train_loss_tracker = keras.metrics.Mean(name='train_loss')

    senet.compile(optimizer=optimizer_fn,loss=loss_mse,metrics=CategoricalAccuracy())
   
    strat_time = time()
    train_datagen,test_datagen = make_data_generator(args.test)
    #for step in range(4):
        #optimizer_fn.learning_rate = learning_rates[step]
        
        #print(f'Starting training step {step + 1}')
        #print(optimizer_fn.learning_rate)
    for epoch in range(epochs):
        print(f'epoch {epoch+1} / {epochs}')
        train(
            model=senet,generator=train_datagen,
            optimizer=optimizer_fn)

        #if validation:
            #va_acc = test(model=senet,generator=val_datagen)
            #print(f'validation acc = {va_acc:.4f}')
            #if va_acc > best_val_acc:
                #best_val_acc = va_acc

                #te_acc = test(model=senet,generator=test_datagen)
                #if te_acc > best_test_acc:
                    #best_test_acc = te_acc
                    #senet.save_weights(weights_h5)
                    #senet.save(model_h5)
                #print(f'best test accuracy: {best_test_acc:.4f}')

        #else:
        te_acc = test(model=senet,generator=test_datagen)
        if te_acc > best_test_acc:
            best_test_acc = te_acc
            #senet.save_weights(weights_h5)
            senet.save(model_h5)
        print(f'test accuracy: {te_acc:.4f}')
        print(f'best test accuracy: {best_test_acc:.4f}')
        
            #learning rate scheduling

        #senet.load_weights(weights_h5)
        #optimizer_fn.learning_rate = optimizer_fn.learning_rate * 0.5
    
    print('Multi Step Training has just been ended')
    end_time = time() - strat_time
    print(f'trainig time: {end_time}')
    print(f'best test accuracy: {best_test_acc:.4f}')
    loaded_model = keras.models.load_model(
        model_h5,
        custom_objects={
            'Weighted_Euclidean_Distance':Weighted_Euclidean_Distance,
            'BilinearInterpolation':BilinearInterpolation,
            'Localization':Localization},compile=False)
    inp = keras.layers.Input((dim,dim,3))
    enc = keras.Model(
        inputs=loaded_model.get_layer('encoder').input,
        outputs=loaded_model.get_layer('encoder').output)
    enc.save(encoder_h5)
    print(f'cascade encoder saved at {encoder_h5}')
    
    