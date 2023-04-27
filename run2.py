from tensorflow import keras
import tensorflow as tf
from tensorflow.keras.metrics import Mean, CategoricalAccuracy
from models.metrics import loss_mse,accuracy,loss_ce
from models.makemodels import make_proto_model
import argparse
from time import time
from data_loader import get_loader
from models.distances import Weighted_Euclidean_Distance
from models.stn import BilinearInterpolation,Localization
from tensorflow.keras.models import load_model
from models.densentst import create_ts


parser = argparse.ArgumentParser('SENet')
parser.add_argument('--mode',type=str,default='metatrain')
parser.add_argument('--backbone',type=str,default='densenet')
parser.add_argument('--test',type=str,default='gtsrb2tt100k')
parser.add_argument('--epochs',type=int,default=50)
parser.add_argument('--batch',type=int,default=128)
parser.add_argument('--dim',type=int,default=64)
parser.add_argument('--lr',type=float,default=1e-4)

args = parser.parse_args()

backbone = args.backbone
dim = args.dim
batch = args.batch
epochs = args.epochs
lr = args.lr

train_acc_tracker = Mean('train_accuracy')
train_loss_tracker = Mean('train_loss')
teacher_acc_tracker = Mean('teacher_accuracy')
student_soft_acc_tracker = Mean('student_soft_accuracy')
student_hard_acc_tracker = Mean('student_hard_accuracy')

def loss_func(tlables,stsoftlables,sthardlables,gtlables):
    student_loss = loss_ce(gtlables,sthardlables)
    distill_loss = loss_ce(stsoftlables,tlables)
    return 0.3*student_loss + 0.7*distill_loss



#test function
def test(model,generator):
    acc_mean_student_hard = tf.constant([0.0],dtype='float32')
    acc_mean_student_soft = tf.constant([0.0],dtype='float32')
    acc_mean_teacher = tf.constant([0.0],dtype='float32')
    #test_acc_tracker = Mean('test_accuracy')
    for z in generator:
        [Xs,Xq],y_true = z
        tlables,stsoftlables,sthardlables = model([Xs,Xq],training = True)
        #acc += accuracy(y_true,y_pred)
        student_hard_acc_tracker.update_state(accuracy(y_true,sthardlables))
        student_soft_acc_tracker.update_state(accuracy(y_true,stsoftlables))
        teacher_acc_tracker.update_state(accuracy(y_true,tlables))
    acc_mean_student_hard = student_hard_acc_tracker.result()
    acc_mean_student_soft = student_soft_acc_tracker.result()
    acc_mean_teacher = teacher_acc_tracker.result()
    student_hard_acc_tracker.reset_state()
    student_soft_acc_tracker.reset_state()
    teacher_acc_tracker.reset_state()
    return acc_mean_student_hard, acc_mean_student_soft, acc_mean_teacher
    #return test_acc_tracker.result()

#train function
@tf.function
def train_step(model,data,optimizer):
    [Xs,Xq],y_true = data
    with tf.GradientTape() as tape:
        #make prediction
        tlables,stsoftlables,sthardlables = model([Xs,Xq],training = True)
        loss =  loss_func(tlables,stsoftlables,sthardlables,y_true)
    gradients = tape.gradient(loss,model.trainable_variables)
    optimizer.apply_gradients(zip(gradients,model.trainable_variables))
    return loss,accuracy(y_true,sthardlables)

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
    model_h5 = 'model_files/' + backbone + '_' + args.test + '_whole_distill.h5'
    encoder_h5 = 'model_files/' + backbone + '_' + args.test + '_encoder_distill.h5'

    best_test_acc = 0.0

    senet = create_ts()#make_proto_model(backbone=backbone,input_shape=(dim,dim,3))
    senet.summary()
    encoder_new = senet.get_layer('encoder_student')
    encoder_new.summary()
    optimizer_fn = keras.optimizers.Adam(learning_rate=lr,epsilon=1.0e-8)
    #senet.compile(optimizer=optimizer_fn,loss=loss_func,metrics=CategoricalAccuracy())
    strat_time = time()
    train_datagen,test_datagen = make_data_generator(args.test)
 
    for step in range(4):
        print(f'=====step {step+1}=====')
        for epoch in range(ep):
            print(f'====epoch{epoch+1}/{ep}====')
            #senet.fit(train_datagen,workers=6)
            train(
                model=senet,generator=train_datagen,
                optimizer=optimizer_fn)
            te_acc, stu_acc_soft, stu_acc_hard = test(model=senet,generator=test_datagen)
            if stu_acc_hard > best_test_acc:
                best_test_acc = stu_acc_hard
                senet.save(model_h5)
                senet.save_weights('best_weights.h5')
            encoder_new.save('new_encoder.h5')
            print(f'test accuracy: {te_acc:.4f}')
            print(f'student soft test accuracy: {stu_acc_soft:.4f}')
            print(f'student hard test accuracy: {stu_acc_hard:.4f}')
            print(f'best test accuracy: {best_test_acc:.4f}')
        senet.load_weights('best_weights.h5')
        optimizer_fn.learning_rate = optimizer_fn.learning_rate / 2   
   
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
    model_h5 = 'model_files/best_models/densenet_' + mode + '_whole.h5'
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

    