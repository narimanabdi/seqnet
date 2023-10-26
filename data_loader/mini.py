from .batchgenerator import MINI_Generator

def get_generator(batch=128,dim=64):
    tr_gen = MINI_Generator(n_way=5,k_shot=1,qnum=15,data_type='train',target_size=(dim,dim),episode=1000)
    te_gen = MINI_Generator(n_way=5,k_shot=1,qnum=15,data_type='test',target_size=(dim,dim),episode=400)
    return tr_gen,te_gen

def get_test_generator(batch=128,dim=64):
    return MINI_Generator(n_way=5,k_shot=1,data_type='test',target_size=(dim,dim))