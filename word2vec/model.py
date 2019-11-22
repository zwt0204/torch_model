# -*- encoding: utf-8 -*-
"""
@File    : model.py
@Time    : 2019/11/22 15:24
@Author  : zwt
@git   : https://github.com/Adoni/word2vec_pytorch/
@Software: PyCharm
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class SkipGramModel(nn.Module):

    def __init__(self, emb_size, emb_dimension, batch_size, window_size, iteration, initial_lr, min_count):
        super(SkipGramModel, self).__init__()
        # 词个数
        self.emb_size = emb_size
        # 维度
        self.emb_dimension = emb_dimension
        # 批次
        self.batch_size = batch_size
        # 窗口大小
        self.window_size = window_size
        self.iteration = iteration
        # 学习率
        self.initial_lr = initial_lr
        # 最少出现次数
        self.min_count = min_count
        # 中心词
        self.u_embeddings = nn.Embedding(self.emb_size, self.emb_dimension, sparse=True)
        # 周边词
        self.v_embeddings = nn.Embedding(self.emb_size, self.emb_dimension, sparse=True)
        initrange = 0.5 / self.emb_dimension
        self.u_embeddings.weight.data.uniform_(-initrange, initrange)
        self.v_embeddings.weight.data.uniform_(-0, 0)

    def forward(self, pos_u, pos_v, neg_v):
        """Forward process.
        As pytorch designed, all variables must be batch format, so all input of this method is a list of word id.
        Args:
            pos_u: list of center word ids for positive word pairs.
            pos_v: list of neibor word ids for positive word pairs.
            neg_u: list of center word ids for negative word pairs.
            neg_v: list of neibor word ids for negative word pairs.
        Returns:
            Loss of this process, a pytorch variable.
        """
        emb_u = self.u_embeddings(pos_u)
        emb_v = self.v_embeddings(pos_v)
        score = torch.mul(emb_u, emb_v).squeeze()
        score = torch.sum(score, dim=1)
        score = F.logsigmoid(score)
        neg_emb_v = self.v_embeddings(neg_v)
        neg_score = torch.bmm(neg_emb_v, emb_u.unsqueeze(2)).squeeze()
        neg_score = F.logsigmoid(-1 * neg_score)
        return -1 * (torch.sum(score) + torch.sum(neg_score))

    def save_embedding(self, id2word, file_name, use_cuda):
        """Save all embeddings to file.
        As this class only record word id, so the map from id to word has to be transfered from outside.
        Args:
            id2word: map from word id to word.
            file_name: file name.
        Returns:
            None.
        """
        if use_cuda:
            embedding = self.u_embeddings.weight.cpu().data.numpy()
        else:
            embedding = self.u_embeddings.weight.data.numpy()
        fout = open(file_name, 'w', encoding='utf8')
        fout.write('%d %d\n' % (len(id2word), self.emb_dimension))
        for wid, w in id2word.items():
            e = embedding[wid]
            e = ' '.join(map(lambda x: str(x), e))
            fout.write('%s %s\n' % (w, e))
