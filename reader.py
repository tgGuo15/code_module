import pickle
import os.path as osp
import threading
import numpy as np
import random
class Reader(object):
    def __init__(self,mode,task,model_mode,
                 path='dataset/torchDataset',
                 action_path='dataset/action.pkl',
                 pos_path='dataset/pos.pkl'):
        self.mode = mode
        self.model_mode = model_mode
        self.task = task
        file_name = osp.join(path,mode+'_set.pkl')
        self.train_action, self.valid_action, self.test_action = pickle.load(open(action_path))
        self.train_pos, self.valid_pos, self.test_pos = pickle.load(open(pos_path))
        # data_x (number), x_data (real word), data_y (label)
        self.dataset = pickle.load(open(file_name))
        self.task_transform(task)
    def task_transform(self,task):
        self.data = []
        self.pos = []
        self.action = []
        self.label = []
        self.sentence = []
        self.length = []
        if task == 2: # binary
            for i in range(len(self.dataset[2])):
                if self.dataset[2][i] >= 3:
                    self.label.append(1)
                    self.data.append(self.dataset[1][i])
                    self.sentence.append(self.dataset[0][i])
                    self.length.append(self.dataset[3][i])
                    if self.mode == 'train':
                        self.pos.append(self.train_pos[i])
                        self.action.append(self.train_action[i])
                    elif self.mode == 'valid':
                        self.pos.append(self.valid_pos[i])
                        self.action.append(self.valid_action[i])
                    elif self.mode == 'test':
                        self.pos.append(self.test_pos[i])
                        self.action.append(self.test_action[i])
                elif self.dataset[2][i] <= 1:
                    self.label.append(0)
                    self.data.append(self.dataset[1][i])
                    self.sentence.append(self.dataset[0][i])
                    self.length.append(self.dataset[3][i])
                    if self.mode == 'train':
                        self.pos.append(self.train_pos[i])
                        self.action.append(self.train_action[i])
                    elif self.mode == 'valid':
                        self.pos.append(self.valid_pos[i])
                        self.action.append(self.valid_action[i])
                    elif self.mode == 'test':
                        self.pos.append(self.test_pos[i])
                        self.action.append(self.test_action[i])
        elif task == 5:
            self.sentence, self.data, self.label, self.length = self.dataset
            if self.mode == 'train':
                self.action = self.train_action
                self.pos = self.train_pos
            elif self.mode == 'valid':
                self.action = self.valid_action
                self.pos = self.valid_pos
            elif self.mode == 'test':
                self.action = self.test_action
                self.pos = self.test_pos
        self.length = np.array(self.length)
        self.max_length = max(self.length)
        self.min_length = min(self.length)
        self.len_bucket = {}
        for k in range(self.min_length,self.max_length+1):
            index = np.where(self.length == k)[0]
            self.len_bucket[k] = list(index)
        self.n_count = self.min_length
        self.index = range(len(self.length))
        self.num_count = 0
        random.shuffle(self.index)

    def get_batch(self):
        if self.model_mode == 'bilstm':
            label = []
            data = np.ones((self.n_count,len(self.len_bucket[self.n_count])),dtype=np.int64)
            for k in range(len(self.len_bucket[self.n_count])):
                data[:,k] = self.data[self.len_bucket[self.n_count][k]]
                label.append(self.label[self.len_bucket[self.n_count][k]])
            label = np.array(label)
            self.n_count += 1
            if self.n_count > self.max_length:
                self.n_count = self.min_length
            return data,label
        elif self.model_mode == 'tree':
            data = self.data[self.index[self.num_count]]
            label = self.label[self.index[self.num_count]]
            action = self.action[self.index[self.num_count]]
            pos = self.pos[self.index[self.num_count]]
            self.num_count += 1
            if self.num_count == len(self.length):
                random.shuffle(self.index)
                self.num_count = 0
            return np.array([data]).transpose(), np.array([label]), action, pos






