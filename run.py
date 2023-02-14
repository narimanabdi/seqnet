from tensorflow import keras
import tensorflow as tf
from tensorflow.keras.metrics import Mean, CategoricalAccuracy
from models.metrics import loss_mse,accuracy
from models.makemodels import make_proto_model
import argparse
from time import time
from data_loader import get_loader
from models.distances import Weighted_Euclidean_Distance
from models.stn import BilinearInterpolation,Localization
from tensorflow.keras.models import load_model


parser = argparse.ArgumentParser('SENet')
#parser.add_argument('--cfg',type = str,required = True,help = 'config file')
parser.add_argument('--mode',type=str,default='metatrain')
parser.add_argument('--backbone',type=str,default='densenet')
parser.add_argument('--test',type=str,default='gtsrb2tt100k')
parser.add_argument('--epochs',type=int,default=50)
parser.add_argument('--batch',type=int,default=128)
parser.add_argument('--dim',type=int,default=64)
parser.add_argument('--lr',type=float,default=1e-4)
parser.add_argument('--finetune',type = str,default='no')

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

def meta_train(ep):
    model_h5 = 'model_files/' + backbone + '_' + args.test + '_whole.h5'
    encoder_h5 = 'model_files/' + backbone + '_' + args.test + '_encoder.h5'

    best_test_acc = 0.0

    senet = make_proto_model(backbone=backbone,input_shape=(dim,dim,3))
    optimizer_fn = keras.optimizers.Adam(learning_rate=lr,epsilon=1.0e-8)
    senet.compile(optimizer=optimizer_fn,loss=loss_mse,metrics=CategoricalAccuracy())
    strat_time = time()
    train_datagen,test_datagen = make_data_generator(args.test)
    plato = 0
    for step in range(2):
        print(f'=====step {step+1}=====')
        for epoch in range(ep):
            print(f'====epoch{epoch+1}/{ep}====')
            train(
                model=senet,generator=train_datagen,
                optimizer=optimizer_fn)
            te_acc = test(model=senet,generator=test_datagen)
            if te_acc > best_test_acc:
                best_test_acc = te_acc
                senet.save(model_h5)
                senet.save_weights('best_weights.h5')
            print(f'test accuracy: {te_acc:.4f}')
            print(f'best test accuracy: {best_test_acc:.4f}')
        senet.load_weights('best_weights.h5')
        #optimizer_fn.learning_rate = optimizer_fn.learning_rate / 2   
        optimizer_fn.learning_rate = 2.5e-5
        ep = 150     
    print('Meta Training has just been ended')
    end_time = time() - strat_time
    print(f'trainig time: {end_time}')
    print(f'best test accuracy: {best_test_acc:.4f}')
    loaded_model = load_model(
        model_h5,
        custom_objects={
            'Weighted_Euclidean_Distance':Weighted_Euclidean_Distance,
            'BilinearInterpolation':BilinearInterpolation,
            'Localization':Localization},compile=False)
    enc = keras.Model(
        inputs=loaded_model.get_layer('encoder').input,
        outputs=loaded_model.get_layer('encoder').output)
    enc.save(encoder_h5)
    print(f'cascade encoder saved at {encoder_h5}')

def meta_test(mode):
    loader = get_loader(mode) 
    test_generator = loader.get_test_generator(batch=batch,dim=64)
    if args.finetune == 'yes':
        model_h5 = 'model_files/best_models/densenet_' + args.test + '_whole_ft.h5'
    else:
        model_h5 = 'model_files/best_models/densenet_' + args.test + '_whole.h5'
    senet = load_model(
        model_h5,
        custom_objects={
            'Weighted_Euclidean_Distance':Weighted_Euclidean_Distance,
            'BilinearInterpolation':BilinearInterpolation,
            'Localization':Localization},compile=False)

    
    metric = CategoricalAccuracy()
    senet.compile(metrics=metric)
     
    best_acc = 0
    for ite in range(5):
        print(f'\033[0;36mstrating test iteration {ite+1}\033[0m')
        _,acc = senet.evaluate(test_generator,verbose=1)
        if acc > best_acc:
            best_acc = acc
    print('\033[0;31m')
    print('+---------------------------+')
    print('|       Meta Testing Report      |')
    print('+---------------------------+')
    print(f'|      Test    |{args.test}|')
    print('+---------------------------+')
    print(f'|Best accuracy |{best_acc:.4f}     |')
    print('+---------------------------+')
    print('\033[0m')


if __name__ == "__main__":

    if args.mode == 'metatrain':
        meta_train(args.epochs)
    if args.mode == 'metatest':
        meta_test(mode = args.test)

    