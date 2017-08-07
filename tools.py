import time
import logging
import os.path as osp
import numpy as np
class Timer:
    def __init__(self):
        self.start_time = 0
        self.end_time = 0
        self.total_time = 0
        self.avg_time = 0
        self.n_toc = 0

    def tic(self):
        self.n_toc = 0
        self.start_time = time.time()

    def toc(self):
        self.end_time = time.time()
        self.total_time = self.end_time - self.start_time
        self.n_toc += 1.
        self.avg_time = self.total_time / self.n_toc
        return self.total_time

class Logger:
    """
    When receiving a message, first print it on screen, then write it into log file.
    If save_dir is None, it writes no log and only prints on screen.
    """
    def __init__(self, save_dir):
        if save_dir is not None:
            self.logger = logging.getLogger()
            logging.basicConfig(filename=osp.join(save_dir, 'experiment.log'), format='%(asctime)s |  %(message)s')
            logging.root.setLevel(level=logging.INFO)
        else:
            self.logger = None

    def info(self, msg, to_file=True):
        print msg
        if self.logger is not None and to_file:
            self.logger.info(msg)

def get_dropout(data,label,rate,size):
    seq_len = data.shape[0]
    temp = np.zeros((seq_len,size),dtype=np.int64)
    for i in range(size):
        temp[:,i] = data[:,0]
    e_p = np.random.uniform(0, 1, size=(temp.shape[0],size-1))
    [index1, index2] = np.where(e_p < rate)
    for jj in range(len(index1)):
        temp[index1, index2] = 0  # <unk>
    label = np.array([label[0]]* size)
    return temp,label
