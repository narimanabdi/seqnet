##################################
# Generator for GTSRB --> TT100K #
##################################
from .batchgenerator import GTSRB_Generator,TT100K_Generator

def get_generator(batch=128,dim=64):
    tr_gen = GTSRB_Generator(
        n_way=43,k_shot=1,batch=batch,data_type='all',
        target_size=(dim,dim))
    te_gen = TT100K_Generator(
        n_way=36,k_shot=1,batch=batch,data_type='all',target_size=(dim,dim))
    return tr_gen,te_gen

def get_test_generator(batch=128,dim=64,type='all'):
    if type == 'all':
        return TT100K_Generator(
            n_way=36,k_shot=1,batch=batch,data_type='all',target_size=(dim,dim))
    else:
        return TT100K_Generator(
            n_way=32,k_shot=1,batch=batch,data_type='unseen',target_size=(dim,dim))