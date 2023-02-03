# coding=utf-8
import json
import pickle as pkl


class DataProcess:
    def __init__(self):
        pass

    def data_read(self, data_path, data_save_path, vocab_path=None):
        data = json.load(open(data_path, mode='r', encoding='utf-8'))  # 加载原始json文件
        data_save = open(data_save_path, mode='wb')  # 保存处理之后的数据集
        qa = []
        question_vocab = dict()  # 问题字典
        for tmp_data in data:
            """
            tmp_data是原始数据，样例如下:
             {
                "Id": 1,
                "Question": "what is the end time for the daily show as rachael harris has cast member",
                "Temporal signal": [
                    "No signal"
                ],
                "Temporal question type": [
                    "Temp.Ans"
                ],
                "Answer": [
                    {
                        "AnswerType": "Value",
                        "AnswerArgument": "2003-03-20T00:00:00Z"
                    }
                ],
                "Data source": "LC-QuAD 2.0 (ISWC 2019)",
                "Question creation date": "2019-01-23",
                "Data set": "train"
            }
               
            """
            question = tmp_data['Question'].strip().replace('\n', '')  # 提取问题
            answer = tmp_data['Answer']  # 提取答案
            answer_all = []
            if len(answer) > 0:
                for k in answer:
                    # if 'WikidataLabel' in k:
                    if k['AnswerType'].__eq__('Entity'):  # 实体答案
                        answer_all.append(k['WikidataQid'])  # 获取Entity对应的：QID
                    elif k['AnswerType'].__eq__('Value'):  # 非实体答案
                        answer_all.append(k['AnswerArgument'])  # 获取Value对应的：Value
                    else:
                        pass  # 其它答案
                if len(answer_all) > 0:  # 过滤掉无答案的问题
                    qa.append([question, answer_all])  # 形成一条记录
                    # vocab
                    question_split = question.split(' ')  # 问题分词，制作问题字典
                    for cur_word in question_split:
                        cur_word = cur_word.lower()  # 分词统一小写，减少词向量个数
                        if cur_word not in question_vocab.keys():  # 字典
                            question_vocab[cur_word] = len(question_vocab)

        pkl.dump(qa, data_save)
        if vocab_path is not None:  # 如果保存字典
            question_word_save = open(vocab_path, mode='wb')
            pkl.dump(question_vocab, question_word_save)  # 二进制的方式保存（好处是，按源格式保存）
            print('word num:', len(question_vocab.keys()))
            print('all word:', list(question_vocab.keys()))


if __name__ == "__main__":
    f = DataProcess()
    # train.json: 原始文件
    # 读取原始文件，并保存之后形成：train.pkl
    # 读取原始文件过程中，形成的问题字典：question_vocab.pkl
    f.data_read('../../Source/train.json', '../../processed/train.pkl', '../../processed/question_vocab.pkl')
    f.data_read('../../Source/test.json', '../../processed/test.pkl')
    f.data_read('../../Source/dev.json', '../../processed/dev.pkl')