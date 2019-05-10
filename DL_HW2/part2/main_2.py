import torch 
import numpy as np
from model_2 import RNNModel
import pickle
import tensorboardX

idx2word=pickle.load(open('idx2word.p','rb'))
word2idx=pickle.load(open('word2idx.p','rb'))

INPUTDIM = 15
OUTPUTDIM= 2
HIDDENDIM= 30
VOCABSIZE= max(word2idx.values())
NUMOFLAYER= 2

EPOCH = 40
LR = 1e-3
BATCH_SIZE = 15




if __name__ == "__main__":

    # all_data = pickle.load(open('all_corpus.p','rb'))
    true_data = pickle.load(open('true_data.p','rb'))
    false_data = pickle.load(open('false_data.p','rb'))

    train_data = true_data[45:] + false_data[55:]
    test_data = true_data[:45] + false_data[:55]

    print(len(true_data))
    print(len(false_data))

    seq_train_loader = prepro_iter(train_data,'word2idx.p')
    seq_val_loader   = prepro_iter(test_data,'word2idx.p')

    for epoch in range(EPOCH):
        seq_train_loader.epoch_iter()
        iter_ = seq_train_loader.batch_iter(BATCH_SIZE)
        for batch_seq,batch_len,batch_label in  iter_:
            

            
            




    

    





