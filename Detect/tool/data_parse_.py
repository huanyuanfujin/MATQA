# coding=utf-8
import re
import os
import csv
import sys
import pickle
import numpy as np
from random import sample


class DataParse:
    def __init__(self, train_data_path, task2id_id2task_path, sentence_task_path, vocb_path):

        self.train_data_path = train_data_path  # 训练数据路径
        self.task2id_id2task = task2id_id2task_path  # 任务的映射
        self.sentence_task_path = sentence_task_path  # 训练数据
        self.vocb_path = vocb_path  # 字典
        self.stop = ['”', '“', '、', '。', '，', '──', '……', '（', '）', '？', '《', '》', '<', '>',
                     '！', '......', '.', ',', '；', ';', '%']

    @property
    def stopwords(self):
        return self.stop

    def read_corpus(self, corpus_path=None, save=True):
        """
            获取语料，路径
        """
        # if corpus_path is None:
        #     assert 'path empty'
        try:
            task_sentense = []
            task2id_id2task = {}
            task2id = {}
            id2task = {}
            skip_title = True
            lines = csv.reader(open(corpus_path, mode='r', encoding='utf-8', errors='ignore'))
            for line in lines:
                if skip_title:
                    skip_title = False
                else:
                    content = list(''.join(re.findall('[\u4e00-\u9fa5]+', line[1].strip().split(' ')[0])))
                    content = list(filter(lambda y: len(y) > 0, content))
                    comment_all = list(map(lambda y: ''.join(re.findall('[\u4e00-\u9fa5]+', y)), line[8].split('\t')))
                    comment_all = list(filter(lambda y: len(y) > 0, comment_all))
                    if len(comment_all) > 1:
                        sentence = []
                        for k in comment_all:
                            k = list(filter(lambda y: len(y) > 0, list(k)))
                            if len(k) > 0:
                                sentence.append(k)
                        sentence.insert(0, content)
                        task = line[5].strip()  # 标签
                        print(task, sentence)
                        task_sentense.append((sentence, [task]))  # 句子-内容
                        for cur_task in [task]:
                            if task2id.get(cur_task, 10000) == 10000:
                                task2id[cur_task] = len(task2id)
                                id2task[len(id2task)] = cur_task

            task2id_id2task['task2id'] = task2id
            task2id_id2task['id2task'] = id2task
            print(task2id_id2task)
            if save:
                pickle.dump(task_sentense, open(self.sentence_task_path, mode='wb'))
                pickle.dump(task2id_id2task, open(self.task2id_id2task, mode='wb'))
            return task_sentense, task2id_id2task
        except Exception:
            raise Exception

    def creat_vocab(self, data=None, save=True):
        # if not os.path.exists(self.vocb_path):
        if True:
            vocab = dict()  # 创建字典
            vocab['<UNK>'] = 1  # 未知的词汇
            vocab['<PAD>'] = 0  # 需要被填充的标记
            if data is not None:
                assert isinstance(data, list) and isinstance(data[0], tuple)
                for task_sentence in data:
                    all_splited = task_sentence[0]  # 得到任务 和 句子
                    print('类别-句子', all_splited)
                    for cut_word in all_splited:
                        for jjj in cut_word:
                            if vocab.get(jjj, -1) == -1:
                                vocab[jjj] = len(vocab)
                if save:
                    print('vocab:', vocab)
                    pickle.dump(vocab, open(self.vocb_path, mode='wb'))
                return vocab
            else:
                print('data empty......')
        else:
            sys.stdout.write('vocab exists......')
            return pickle.load(open(self.vocb_path, mode='rb'))

    def random_embedding(self, embedding_dim, word_num):
        """
        随机的生成word的embedding，这里如果有语料充足的话，可以直接使用word2vec蓄念出词向量，这样词之间的区别可能更大。
        :param embedding_dim:  词向量的维度。
        :return: numpy format array. shape is : (vocab, embedding_dim)
        """
        # if vocb_paths is None:
        #     vocab_creatation = pickle.load(open(self.vocb_path, mode='rb'))
        # else:
        #     vocab_creatation = pickle.load(open(vocb_paths, mode='rb'))
        """
        one-hot
        0 0 0 0 0 0 0 0 0 1 0 0 ... 0
        正太分布
        10 ：128维
        625 ： 625 * 128
        """
        embedding_mat = np.random.uniform(-0.25, 0.25, (word_num, embedding_dim))  # 正太分布  '次': 10
        embedding_mat = np.float32(embedding_mat)
        return embedding_mat


class Data_Inter:
    """
    生成训练数据
    """
    def __init__(self, batch_size, task_sentence_path, intent2id, vocb_path):
        self.task_sentence_path = task_sentence_path  # 任务—句子
        self.intent2id = pickle.load(open(intent2id, mode='rb'))['task2id']  # intent的映射
        # print('self.intent2id test:', self.intent2id)
        self.vocb_path = vocb_path  # 字典路径
        self.batch_size = batch_size  # 批的大小
        self.train = np.array(pickle.load(open(self.task_sentence_path[0], mode='rb')))
        self.train_end = len(self.train)
        self.shuffle_train = sample(range(0, self.train_end, 1), self.train_end)
        self.test = np.array(pickle.load(open(self.task_sentence_path[1], mode='rb')))
        self.test_end = len(self.test)
        self.shuffle_test = sample(range(0, self.test_end, 1), self.test_end)

        self.train_index = 0
        self.test_index = 0

        print('data info\n')
        print('\t train:', self.train_end)
        print('\t test:', self.test_end)

        if os.path.exists(self.vocb_path):  # 读取字典
            self.vocab = pickle.load(open(self.vocb_path, mode='rb'))
        else:
            print('vocab is empty......')

    def next(self, seq_length, if_tcn=False):  # 训练样本迭代
        sentence = []
        sentence_length = []
        real_intent = []
        if self.train_index + self.batch_size <= self.train_end:
            it_data = self.shuffle_train[self.train_index: self.train_index + self.batch_size]  # 迭代数据
            self.train_index += self.batch_size
        elif self.train_index + self.batch_size == self.train_end:
            it_data = self.shuffle_train[self.train_index: self.train_end]
            self.shuffle_train = sample(range(0, self.train_end, 1), self.train_end)
            self.train_index = 0

        else:
            it_data = self.shuffle_train[self.train_index: self.train_end]  # 随机选取
            self.shuffle_train = sample(range(0, self.train_end, 1), self.train_end)
            remain = self.shuffle_train[0: self.train_index + self.batch_size - self.train_end]  # 剩余
            self.train_index = 0
            it_data = np.concatenate((it_data, remain), axis=0)
        batch_sentence = self.train[it_data, :]
        # print('batch_sentence Multi:', batch_sentence[0])

        for cur_sentences_index, cur_sentences in enumerate(batch_sentence):
            tmp = self.sentence2index(cur_sentences[0] + ["<PAD>"] * max(0, seq_length - len(cur_sentences[0])), self.vocab)
            # print('self.intent2id:', self.intent2id)

            real_intent.append(self.intent2id[cur_sentences[1]])
            sentence.append(tmp)
            sentence_length.append(len(cur_sentences[0]))
        # print('np.array(task_intent):', np.array(task_intent).shape)
        return np.array(sentence), real_intent, sentence_length

    def next_test(self, seq_length, if_tcn=False):  # 测试集上的迭代
        sentence = []
        sentence_length = []
        task_intent = []
        real_intent = []
        if self.test_index + self.batch_size <= self.test_end:
            it_data = self.shuffle_test[self.test_index: self.test_index + self.batch_size]  # 迭代数据
            self.test_index += self.batch_size
        elif self.test_index + self.batch_size == self.test_end:
            it_data = self.shuffle_test[self.test_index: self.test_end]
            self.shuffle_test = sample(range(0, self.test_end, 1), self.test_end)
            self.test_index = 0

        else:
            it_data = self.shuffle_test[self.test_index: self.test_end]  # 随机选取
            self.shuffle_test = sample(range(0, self.test_end, 1), self.test_end)
            remain = self.shuffle_test[0: self.test_index + self.batch_size - self.test_end]  # 剩余
            self.test_index = 0
            it_data = np.concatenate((it_data, remain), axis=0)
        batch_sentence = self.test[it_data, :]
        # print('batch_sentence Multi:', batch_sentence[0])

        for cur_sentences_index, cur_sentences in enumerate(batch_sentence):
            tmp = self.sentence2index(cur_sentences[0] + ["<PAD>"] * max(0, seq_length - len(cur_sentences[0])), self.vocab)
            real_intent.append(self.intent2id[cur_sentences[1]])
            sentence.append(tmp)
            sentence_length.append(len(cur_sentences[0]))
        return np.array(sentence), real_intent, sentence_length,

    def sentence2index(self, sen, vocab):
        assert isinstance(sen, list) and len(sen) > 0
        assert isinstance(vocab, dict) and len(vocab) > 0
        sen2id = []
        for cur_sen in sen:
            sen2id.append(vocab.get(cur_sen, 0))  # 如果找不到，就用0代替。
        return sen2id

    def task2index(self, cur_tasks, mapping):
        assert isinstance(cur_tasks, list) and len(cur_tasks) > 0 and hasattr(cur_tasks, '__len__')
        assert isinstance(mapping, dict) and len(mapping) > 0
        cur_task2index_mapping = []
        for cur_task in cur_tasks:
            if cur_task in mapping.keys():
                cur_task2index_mapping.append(mapping[cur_task])
            elif 'O' in mapping:
                cur_task2index_mapping.append(mapping['O'])
            else:
                cur_task2index_mapping.append(mapping['<PAD>'])
        return cur_task2index_mapping