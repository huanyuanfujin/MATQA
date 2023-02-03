# coding=utf-8
import pickle
import re
import json


def read_data(path, output_ans_path):
    """
    读取训练集和测试集（path非原始数据集，这是基于原始数据集处理之后的数据）
    :param path:
    :param output_ans_path:
    :return:
    """
    data = pickle.load(open(path, mode='rb'))  # 读取本地文件
    labeled_data_relation_saved = open(output_ans_path, mode='w', encoding='utf-8')  # 另存
    for k in data:
        """
        k的格式:[Question, [ans1 ......ansn]]
        k[0] = Question
        k[1] = [ans1 ......ansn]
        """
        labeled_data_relation_saved.write('\t'.join([k[0].strip(), ','.join(k[1])]) + '\n')
        # print('*' * 100)
        # print(sentence)
        # break


def re_manage_intent_data(path, relation_map):
    data = open(path, mode='r', encoding='utf-8').readlines()
    data_relation = open(relation_map, mode='wb')  # 另存
    relation_data = {'<PAD>': 0}  # 制作答案字典
    for f_data in data:
        """
        k的格式:[Question, [ans1, ......,ansn]]
        k[1] = [ans1, ......,ansn]
        rela = [ans1, ......,ansn]
        """
        print('f_data:', f_data)
        rela = f_data.strip().split('\t')[1].strip().split(',')
        print('rela:', rela)
        for k in rela:
            k = k.strip().lower()  # 5710, 5698
            if k not in relation_data.keys():
                relation_data[k] = len(relation_data)
    print('relation_data:', relation_data)
    print('len, relation_data:', len(relation_data))
    pickle.dump(relation_data, data_relation)


if __name__ == "__main__":
    read_data('../../processed/train.pkl', '../datagenerate/proposed/processed/train_labeled_relation_data.txt')

    read_data('../../processed/test.pkl', '../datagenerate/proposed/processed/test_labeled_relation_data.txt')

    read_data('../../processed/dev.pkl', '../datagenerate/proposed/processed/dev_labeled_relation_data.txt')

    re_manage_intent_data('../datagenerate/proposed/processed/train_labeled_relation_data.txt',
                          '../datagenerate/proposed/data/ans_dict.pkl')
    # import tensorflow
    # print(tensorflow.version.VERSION)
