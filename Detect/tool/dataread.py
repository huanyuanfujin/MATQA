# coding=utf-8
import re
import csv
import json
import pickle
import openpyxl


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
            task2id_id2task = {'task2id': {}, 'id2task': {}}
            skip_title = True
            lines = open(corpus_path, mode='r', encoding='utf-8', errors='ignore').readlines()
            indedx_all = 0
            for line in lines:
                if skip_title:
                    skip_title = False
                else:
                    indedx_all += 1
                    if indedx_all >= 65000:
                        print('indedx_all:', indedx_all)
                        break
                    else:
                        line = line.strip().split("\t")
                        sent = line[0].strip().lower()
                        sent_relation = line[1].strip().lower().split(',')[0]
                        sent = sent.split(' ')
                        task_sentense.append((sent, sent_relation))  # 问题-答案[首个答案，如果答案多余1个，其他答案不考虑，
                        # 算法找到第一个答案之后，基于知识图谱搜索其他答案，答案是否满足要求基于余弦距距离判定# ]
                        print((sent, sent_relation))
                        if sent_relation not in task2id_id2task["task2id"].keys():  # 只考虑首个答案，既包含实体答案和非实体答案
                            task2id_id2task["task2id"][sent_relation] = len(task2id_id2task["task2id"])
                            task2id_id2task["id2task"][len(task2id_id2task["id2task"])] = sent_relation

            if save:
                pickle.dump(task_sentense, open(self.sentence_task_path, mode='wb'))  # 保存问题-首个答案二元组
                pickle.dump(task2id_id2task, open(self.task2id_id2task, mode='wb'))  # 保存首个答案组成的字典
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
                for task_sentence in data:  # 制作问题字典
                    print('task_sentence[0]:', task_sentence[0])
                    for cut_word in task_sentence[0]:  # task_sentence[0]是问题分词
                        if cut_word not in vocab.keys():
                            vocab[cut_word] = len(vocab)
                if save:
                    print('vocab:', vocab)
                    pickle.dump(vocab, open(self.vocb_path, mode='wb'))  # 保存问题节点
                return vocab
            else:
                print('data empty......')
        else:
            sys.stdout.write('vocab exists......')
            return pickle.load(open(self.vocb_path, mode='rb'))


if __name__ == '__main__':

    # so = DataParse(train_data_path=r'../datagenerate/proposed/processed/test_labeled_relation_data.txt',
    #                task2id_id2task_path='../datagenerate/proposed/data/task2id_id2task',
    #                sentence_task_path='../datagenerate/proposed/data/sentence_task_test',
    #                vocb_path='../datagenerate/proposed/data/vocb')
    # so.read_corpus(so.train_data_path, False)
    # so.read_corpus(so.train_data_path, True)
    # row_datas = pickle.load(open(so.sentence_task_path, mode='rb'))
    # so.creat_vocab(row_datas)

    # q511684
    pass