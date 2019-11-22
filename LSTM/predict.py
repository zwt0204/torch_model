# -*- encoding: utf-8 -*-
"""
@File    : predict.py
@Time    : 2019/11/22 14:22
@Author  : zwt
@git   : 
@Software: PyCharm
"""
# coding: utf-8
# 加上from __future__ import print_function这句之后，即使在python2.X，使用print就得像python3.X那样加括号使用。
from __future__ import print_function
import tensorflow.contrib.keras as kr
import torch
from LSTM.cnews_loader import read_category
from LSTM.model import Model_Lstm
import json


class RnnModel:
    def __init__(self):
        self.classes = 10
        self.char_index = {' ': 0}
        self.load_dict()
        self.num_layers = 1
        self.vocab_size = len(self.char_index)
        self.vocab_file = 'data.txt'
        self.drop_out = 0.5
        self.learning_rate = 0.0001
        self.embedding_size = 100
        self.hidden_size = 64
        self.batch_size = 64
        self.sequence = 600
        self.categories, self.cat_to_id = read_category()
        self.model = Model_Lstm(self.classes, self.num_layers, self.vocab_size, self.vocab_file,
                                self.drop_out, self.learning_rate, self.embedding_size, self.hidden_size)
        self.model.load_state_dict(torch.load('model_rnn_params.pkl'))

    def load_dict(self):
        i = 0
        with open(self.vocab_file, "r+", encoding="utf-8") as reader:
            items = json.load(reader)
            for charvalue in items:
                self.char_index[charvalue.strip()] = i + 1
                i += 1

    def predict(self, message):
        content = message
        data = [self.char_index[x] for x in content if x in self.char_index]
        data = kr.preprocessing.sequence.pad_sequences([data], self.sequence)
        data = torch.LongTensor(data)
        y_pred_cls = self.model(data)
        class_index = torch.argmax(y_pred_cls[0]).item()
        return self.categories[class_index]


if __name__ == '__main__':
    model = RnnModel()
    test_demo = [
        '湖人助教力助科比恢复手感 他也是阿泰的精神导师新浪体育讯记者戴高乐报道  上赛季，科比的右手食指遭遇重创，他的投篮手感也因此大受影响。不过很快科比就调整了自己的投篮手型，并通过这一方式让自己的投篮命中率回升。而在这科比背后，有一位特别助教对科比帮助很大，他就是查克·珀森。珀森上赛季担任湖人的特别助教，除了帮助科比调整投篮手型之外，他的另一个重要任务就是担任阿泰的精神导师。来到湖人队之后，阿泰收敛起了暴躁的脾气，成为湖人夺冠路上不可或缺的一员，珀森的“心灵按摩”功不可没。经历了上赛季的成功之后，珀森本赛季被“升职”成为湖人队的全职助教，每场比赛，他都会坐在球场边，帮助禅师杰克逊一起指挥湖人球员在场上拼杀。对于珀森的工作，禅师非常欣赏，“查克非常善于分析问题，”菲尔·杰克逊说，“他总是在寻找问题的答案，同时也在找造成这一问题的原因，这是我们都非常乐于看到的。我会在平时把防守中出现的一些问题交给他，然后他会通过组织球员练习找到解决的办法。他在球员时代曾是一名很好的外线投手，不过现在他与内线球员的配合也相当不错。',
        '弗老大被裁美国媒体看热闹“特权”在中国像蠢蛋弗老大要走了。虽然他只在首钢男篮效力了13天，而且表现毫无亮点，大大地让球迷和俱乐部失望了，但就像中国人常说的“好聚好散”，队友还是友好地与他告别，俱乐部与他和平分手，球迷还请他留下了在北京的最后一次签名。相比之下，弗老大的同胞美国人却没那么“宽容”。他们嘲讽这位NBA前巨星的英雄迟暮，批评他在CBA的业余表现，还惊讶于中国人的“大方”。今天，北京首钢俱乐部将与弗朗西斯继续商讨解约一事。从昨日的进展来看，双方可以做到“买卖不成人意在”，但回到美国后，恐怕等待弗朗西斯的就没有这么轻松的环境了。进展@北京昨日与队友告别  最后一次为球迷签名弗朗西斯在13天里为首钢队打了4场比赛，3场的得分为0，只有一场得了2分。昨天是他来到北京的第14天，虽然他与首钢还未正式解约，但双方都明白“缘分已尽”。下午，弗朗西斯来到首钢俱乐部与队友们告别。弗朗西斯走到队友身边，依次与他们握手拥抱。“你们都对我很好，安排的条件也很好，我很喜欢这支球队，想融入你们，但我现在真的很不适应。希望你们']
    for i in test_demo:
        print(i, ":", model.predict(i))
