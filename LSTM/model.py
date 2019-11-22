# -*- encoding: utf-8 -*-
"""
@File    : model.py
@Time    : 2019/11/22 10:37
@Author  : zwt
参考  : https://blog.csdn.net/weixin_38241876/article/details/90606639
@Software: PyCharm
"""
from torch import nn
import torch.nn.functional as F


class Model_Lstm(nn.Module):
    """
    TextRnn分类
    """
    def __init__(self, classes, num_layers, vocab_size, vocab_file, drop_out, learning_rate, embedding_size, hidden_size):
        super(Model_Lstm, self).__init__()
        self.vocab_size =vocab_size
        self.vocab_file = vocab_file
        self.drop_out = drop_out
        self.learning_rate =learning_rate
        self.embedding_size =embedding_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.classes = classes
        # 输入，词嵌入
        self.embedding = nn.Embedding(self.vocab_size, self.embedding_size)
        # 模型
        self.rnn = nn.LSTM(input_size=self.embedding_size, hidden_size=self.hidden_size, num_layers=self.num_layers
                           , bidirectional=True)
        self.f1 = nn.Sequential(nn.Linear(2 * self.hidden_size, 128),
                                nn.Dropout(self.drop_out),
                                nn.ReLU())
        self.f2 = nn.Sequential(nn.Linear(128, self.classes))

    def forward(self, x):
        x = self.embedding(x)
        x, _ = self.rnn(x)
        x = F.dropout(x, p=self.drop_out)
        x = self.f1(x[:, -1, :])
        return self.f2(x)