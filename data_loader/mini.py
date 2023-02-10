##############################
# Generator for miniImageNet #
##############################
from .batchgenerator import MINI_Generator

def get_generator(batch=128, dim=84):
    tr_gen = MINI_Generator(
        n_way=5,k_shot=1,batch=batch,data_type='seen',
        target_size=(dim,dim))
    te_gen = MINI_Generator(
        n_way=5,k_shot=1,batch=batch,data_type='unseen',target_size=(dim,dim))
    return tr_gen,te_gen

def get_test_generator(batch=1000, dim=64):
    return MINI_Generator(
        n_way=5,k_shot=1,batch=batch,data_type='test',target_size=(dim,dim))