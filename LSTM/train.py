# -*- encoding: utf-8 -*-
"""
@File    : train.py
@Time    : 2019/11/22 10:53
@Author  : zwt
@git   : 
@Software: PyCharm
"""
# 加上from __future__ import print_function这句之后，即使在python2.X，使用print就得像python3.X那样加括号使用。
from __future__ import print_function
import torch
from torch import nn
from torch import optim
from torch.autograd import Variable
import numpy as np
from LSTM.model import Model_Lstm
import json
from LSTM.cnews_loader import read_category, batch_iter, process_file


class Model_Train():

    def __init__(self):
        self.vocab_file = 'D:\gitwork\script\data\dictionary.json'
        self.train_file = 'D:\model\practical-pytorch-master\cnews_data\cnews.train.txt'
        self.classes = 10
        self.char_index = {' ': 0}
        self.load_dict()
        self.num_layers = 1
        self.vocab_size = len(self.char_index)
        self.drop_out = 0.5
        self.learning_rate = 0.0001
        self.embedding_size = 100
        self.hidden_size = 64
        self.batch_size = 64
        self.sequence = 600
        self.model = Model_Lstm(self.classes, self.num_layers, self.vocab_size, self.vocab_file,
                                self.drop_out, self.learning_rate, self.embedding_size, self.hidden_size)

    def load_dict(self):
        i = 0
        with open(self.vocab_file, "r+", encoding="utf-8") as reader:
            items = json.load(reader)
            for charvalue in items:
                self.char_index[charvalue.strip()] = i + 1
                i += 1

    def train(self, epochs):
        # 获取文本的类别及其对应id的字典
        categories, cat_to_id = read_category()
        x_train, y_train = process_file(self.train_file, self.char_index, cat_to_id, self.sequence)
        x_val, y_val = process_file(self.vocab_file, self.char_index, cat_to_id, self.sequence)
        # 损失函数
        Loss = nn.MultiLabelSoftMarginLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        best_val_acc = 0
        for epoch in range(epochs):
            batch_train = batch_iter(x_train, y_train, self.batch_size)
            for x_batch, y_batch in batch_train:
                x = np.array(x_batch)
                y = np.array(y_batch)
                x = torch.LongTensor(x)
                y = torch.Tensor(y)
                x = Variable(x)
                y = Variable(y)
                out = self.model(x)
                loss = Loss(out, y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                acc = np.mean((torch.argmax(out, 1) == torch.argmax(y, 1)).numpy())
                print('epoch %s accracy %s' % (epoch, str(acc)))
            # 验证模型
            if (epoch + 1) % 20 == 0:
                batch_val = batch_iter(x_val, y_val, 100)
                for x_batch, y_batch in batch_val:
                    x = np.array(x_batch)
                    y = np.array(y_batch)
                    x = torch.LongTensor(x)
                    y = torch.Tensor(y)
                    # y = torch.LongTensor(y)
                    x = Variable(x)
                    y = Variable(y)
                    out = self.model(x)
                    loss = Loss(out, y)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    accracy = np.mean((torch.argmax(out, 1) == torch.argmax(y, 1)).numpy())
                    if accracy > best_val_acc:
                        torch.save(self.model.state_dict(), 'model_params.pkl')
                        best_val_acc = accracy
                    print(accracy)


if __name__ == '__main__':
    model_train = Model_Train()
    model_train.train(10)
