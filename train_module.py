from reader import Reader
import pickle
from tools import *
from tree_module import Model
import numpy as np
import os.path as osp
import sys
import os
# glove path
glove_vec_path = 'dataset/glove_vec.pkl'

# task
num_class = 5

train_set = Reader(mode='train',task=num_class,model_mode='tree')
valid_set = Reader(mode='valid',task=num_class,model_mode='tree')
test_set = Reader(mode='test',task=num_class,model_mode='tree')

glove_vec = pickle.load(open(glove_vec_path,'r'))
POS_tag = {}
count = 0
for i in xrange(len(train_set.action)):
    for j in xrange(len(train_set.action[i])):
        if train_set.action[i][j] > 0:
            pos = train_set.pos[i][j]
            if pos not in POS_tag:
                POS_tag[pos] = count
                count += 1

vocab_size = 18282
model = Model(pos_tag=POS_tag,vocab_size=vocab_size,embedding_size=300,hidden_size=128,
              output_size=num_class,lr=0.001,load_glove=True,glove_vec=glove_vec,op_method='Adam')

model.word_embedding.weight.require_grad = False
timer = Timer()
dropout_rate = -1
save_dir = str(num_class)+'/model_result/tree' + 'Adam' + str(dropout_rate)
if not osp.exists(save_dir):
    os.makedirs(save_dir)
logger = Logger(save_dir)
min_num = train_set.min_length
max_num = train_set.max_length

size = 2
Loss = []
best_score = -1.0
timer.tic()
for epoch in range(50000):
    Loss = []
    for k in range(20):
        data, label, action, pos = train_set.get_batch()
        #train_set.num_count = np.random.randint(0,4000)
        if dropout_rate != -1:
           data, label = get_dropout(data,label,dropout_rate,size)
        loss = model.train(data,action,pos,label)
        Loss.append(loss)
    if epoch % 50 == 0:
        acc = 0
        for kk in xrange(len(valid_set.length)):
            data, label, action, pos = valid_set.get_batch()
            if data.shape[1] > 0:
               pred = model.test(data,action,pos).argmax(1)
               err = pred - label
               acc += len(np.where(err ==0)[0])
        score = 1.0 * acc / len(valid_set.length)
        if score > best_score:
            best_score = score
            model.save(osp.join(save_dir,'model.best'))
        model.save(osp.join(save_dir,'model.ckpt'))
        logger.info('[{}], score/best={:.3f}/{:.3f},train loss is {:.6f}, time={:.1f}sec'
                    .format(epoch, score, best_score,np.mean(Loss),timer.toc()))
        timer.tic()
        Loss = []


model.load(osp.join(save_dir,'model.best'))
acc = 0
for kk in xrange(len(test_set.length)):
    data, label, action, pos = test_set.get_batch()
    if data.shape[1] > 0:
       pred = model.test(data,action,label).argmax(1)
       err = pred - label
       acc += len(np.where(err ==0)[0])
score = 1.0 * acc / len(test_set.length)
print('test acc is :',score)
