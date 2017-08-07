import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import os
import pickle
import cPickle

class Model(nn.Module):
   def __init__(self,pos_tag,vocab_size,embedding_size,hidden_size,output_size,lr,load_glove,glove_vec,op_method):
       super(Model, self).__init__()
       self.model = []
       self.embedding_size = embedding_size
       self.hidden_size = hidden_size
       self.output_size = output_size
       self.pos_tag = pos_tag
       self.load_glove = load_glove

       # module network size
       self.module_size = len(pos_tag)
       self.lr = lr
       self.rnn = nn.LSTM(embedding_size,hidden_size/2,num_layers=1,bidirectional=True)
       self.word_embedding = nn.Embedding(vocab_size, embedding_size)

       # initialize glove
       if self.load_glove:
           for i in range(vocab_size):
               self.word_embedding.weight.data[i] = glove_vec[i]

       # transform multi input into single
       self.model = nn.LSTMCell(hidden_size,hidden_size)

       # module network : single FC network
       self.module_network = []
       for j in xrange(self.module_size):
           self.module_network.append(nn.Linear(self.hidden_size,self.hidden_size))
           self.module_network[j] = self.module_network[j].cuda()

       # output layer
       self.output_layer = nn.Linear(hidden_size,output_size)

       if op_method =='Adam':
          self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
       elif op_method == 'SGD':
          self.optimizer = torch.optim.SGD(self.parameters(),lr=self.lr)
       elif op_method == 'RMSprop':
           self.optimizer = torch.optim.RMSprop(self.parameters(), lr=self.lr,weight_decay=0.9)

       self.Relu = nn.ReLU()
       self.Loss = nn.CrossEntropyLoss()

       self.output_layer = self.output_layer.cuda()
       self.cuda()

   # representation of sentence by an Bidirectional LSTM
   def get_word_hidden_state(self,sentence,batch_size):
       # sentence
       x = self.word_embedding(sentence)   # ( seq_length, batch_size, embedding_size)
       h0 = Variable(torch.zeros(2,batch_size,self.hidden_size/2))
       c0 = Variable(torch.zeros(2,batch_size,self.hidden_size/2))
       h0 = h0.cuda()
       c0 = c0.cuda()
       output, hn = self.rnn(x,(h0,c0))
       return output

   # module network
   def forward(self,sentence, action,pos_label,training):
       batch_size = sentence.shape[1]    # always one
       sentence = Variable(torch.LongTensor(sentence.tolist()))
       sentence = sentence.cuda()
       x = self.get_word_hidden_state(sentence,batch_size)  # (seq_length,batch_size,hidden_size)
       stack = []
       count = 0
       for i in xrange(len(action)):
           if action[i] == 0:
               # shift
               stack.append(x[count].view(batch_size, self.hidden_size))
               count += 1
           else:
               # reduce
               h_t = Variable(torch.zeros(batch_size, self.hidden_size))
               c_t = Variable(torch.zeros(batch_size, self.hidden_size))
               h_t = h_t.cuda()
               c_t = c_t.cuda()

               # lstm for transform multi input to fixed
               for j in xrange(action[i]):
                    input = stack.pop()
                    h_t, c_t = self.model(input, (h_t, c_t))  # (batch_size, 100)
               reduce_result = self.module_network[self.pos_tag[pos_label[i]]](h_t)

               stack.append(reduce_result)

       result = self.output_layer(stack[-1])  # ( batch_size, class_num)
       return result

   def train(self,x,action,pos_label,y_true):
       y_true = Variable(torch.LongTensor(y_true.tolist()))
       y_true = y_true.cuda()
       self.optimizer.zero_grad()
       result = self.forward(x,action,pos_label,True)
       loss = self.Loss(result,y_true)
       loss.backward()
       self.optimizer.step()
       return loss.data.cpu().numpy()[0]

   def test(self,x,action,pos_label):
       result = self.forward(x, action,pos_label,False)
       return result.data.cpu().numpy()

   def save(self, file_path):
       with open(file_path, 'wb') as f:
           torch.save(self.state_dict(), f)

   def load(self, file_path):
       with open(file_path, 'rb') as f:
           self.load_state_dict(torch.load(f))
