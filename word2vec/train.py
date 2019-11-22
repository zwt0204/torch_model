# -*- encoding: utf-8 -*-
"""
@File    : train.py
@Time    : 2019/11/22 15:26
@Author  : zwt
@git   : 
@Software: PyCharm
"""
from input_data import InputData
from model import SkipGramModel
from torch.autograd import Variable
import torch
import torch.optim as optim
from tqdm import tqdm


class Word2Vec:
    def __init__(self,
                 input_file_name,
                 output_file_name):
        self.min_count = 5
        self.emb_dimension = 100
        self.batch_size = 64
        self.window_size = 5
        self.iteration = 1
        self.initial_lr = 0.001
        self.data = InputData(input_file_name, self.min_count)
        self.output_file_name = output_file_name
        self.emb_size = len(self.data.word2id)
        self.skip_gram_model = SkipGramModel(self.emb_size, self.emb_dimension, self.batch_size, self.window_size,
                                             self.iteration, self.initial_lr, self.min_count)
        self.use_cuda = torch.cuda.is_available()
        if self.use_cuda:
            self.skip_gram_model.cuda()
        self.optimizer = optim.SGD(
            self.skip_gram_model.parameters(), lr=self.initial_lr)

    def train(self):
        """Multiple training.
        Returns:
            None.
        """
        pair_count = self.data.evaluate_pair_count(self.window_size)
        batch_count = self.iteration * pair_count / self.batch_size
        process_bar = tqdm(range(int(batch_count)))
        for i in process_bar:
            pos_pairs = self.data.get_batch_pairs(self.batch_size,
                                                  self.window_size)
            neg_v = self.data.get_neg_v_neg_sampling(pos_pairs, 5)
            pos_u = [pair[0] for pair in pos_pairs]
            pos_v = [pair[1] for pair in pos_pairs]

            pos_u = Variable(torch.LongTensor(pos_u))
            pos_v = Variable(torch.LongTensor(pos_v))
            neg_v = Variable(torch.LongTensor(neg_v))
            if self.use_cuda:
                pos_u = pos_u.cuda()
                pos_v = pos_v.cuda()
                neg_v = neg_v.cuda()

            self.optimizer.zero_grad()
            loss = self.skip_gram_model.forward(pos_u, pos_v, neg_v)
            loss.backward()
            self.optimizer.step()

            process_bar.set_description("Loss: %0.8f, lr: %0.6f" %
                                        (loss.data,
                                         self.optimizer.param_groups[0]['lr']))
            if i * self.batch_size % 100000 == 0:
                lr = self.initial_lr * (1.0 - 1.0 * i / batch_count)
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = lr
        self.skip_gram_model.save_embedding(
            self.data.id2word, self.output_file_name, self.use_cuda)


if __name__ == '__main__':
    w2v = Word2Vec(input_file_name='zhihu.txt', output_file_name='output.txt')
    w2v.train()