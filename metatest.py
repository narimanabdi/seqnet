from tensorflow import keras
import tensorflow as tf
from tensorflow.keras.layers import TimeDistributed
import numpy as np
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.metrics import CategoricalAccuracy
from data_loader.gtsrb_template_loader import GTSRB_Template_DataGenerator
from data_loader.tt100k_template_loader import TT100K_Template_DataGenerator
from data_loader.persian_template_loader import PERSIAN_Template_DataGenerator
import time
from models import resnet,densenet,conv3b


import argparse
parser = argparse.ArgumentParser('This is test')
parser.add_argument('--backbone',required=True)
parser.add_argument('--episode',default=800,required=True,type=int)
parser.add_argument('--iter',default=1,type=int)
parser.add_argument('--h5',required=True)
parser.add_argument('--target',required=True)

args = parser.parse_args()

if __name__ == '__main__':
    if args.backbone == 'resnet':
        encoder = encoders.create_resnet(input_shape=(64,64,3))
    if args.backbone == 'densenet':
        clf =  densenet.create_model(input_shape = (64,64,3))
    
    
    metric = CategoricalAccuracy()
    clf.compile(metrics=metric)
    model_file = 'model_files/' + args.h5
    clf.load_weights(model_file)
    best_acc = 0
    if args.target == 'tt100k36':
        test_generator = TT100K_Template_DataGenerator(n_way=36,k_shot=1,n_query=100,batch=args.episode,data_type='all',target_size=(64,64))
    elif args.target == 'tt100k32':
        test_generator = TT100K_Template_DataGenerator(n_way=32,k_shot=1,n_query=100,batch=args.episode,data_type='unseen',target_size=(64,64))
    elif args.target == 'pstr':
        test_generator = PERSIAN_Template_DataGenerator(N=38,Ks=1,Kq=1,num_episode=args.episode,data_type='all')
    start_time = time.time()
    for iter in range(args.iter):
        print(f'\033[0;36mstrating test iteration {iter+1} with {args.episode} episodes\033[0m')
        l,acc = clf.evaluate(test_generator,verbose=1)
        if acc >= best_acc:
            best_acc = acc
    print(f'\033[0;31mbest accuracy is {best_acc:.4f}\033[0m')
    print(f'\033[0;32meach episode time: {(time.time() - start_time)/(args.iter*args.episode):.4f}s\033[0m')

