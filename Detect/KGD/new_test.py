# encoding=utf8
import random
import sys
sys.path.append('../')

import re
import json
import pickle
import codecs
import numpy as np
import pickle as pkl
from Detect.proposed.ModelProposed_1 import ProposedModel
import tensorflow as tf
from six import iteritems
from neo_db.query_graph import query
from WordEnhance.model import RNNModel

rnn_layers = 2
embedding_size = 50
hidden_size = 50
input_dropout = 0.5
learning_rate = 0.001
max_grad_norm = 5
num_epochs = 1000
batch_size = 32
seq_length = 10
restore_path = r'../../WordEnhance/model'
all_entity = pkl.load(open(r'../../WordEnhance/all_entity_vec.pkl', mode='rb'))
all_text = open('../../WordEnhance/data/train.txt', mode='r', encoding='utf-8').readlines()
# print('dy_name:', dy_name)


def load_vocab(vocab_file):
    """
    加载字典
    :param vocab_file:
    :return:
    """
    with codecs.open(vocab_file, 'r', encoding='utf-8') as f:
        vocab_index_dict = json.load(f)
    index_vocab_dict = {}
    vocab_size = 0
    for char, index in iteritems(vocab_index_dict):
        index_vocab_dict[index] = char
        vocab_size += 1
    return vocab_index_dict, index_vocab_dict, vocab_size


def cos(a, b):
    """
    余弦相似度
    :param a:
    :param b:
    :return:
    """
    ma = np.linalg.norm(a)
    mb = np.linalg.norm(b)
    # cosine Similarity
    sim = (np.matmul(a, b)) / (ma * mb)
    return sim


def find_lcsubstr_spe(s1, s2_):
    """
    LCS计算S1和S2
    :param s1:
    :param s2_:
    :return:
    """
    all_rrrr = dict()
    for s2 in s2_:
        s2 = s2.strip().replace('\n', '')
        m = [[0 for i in range(len(s2)+1)] for j in range(len(s1)+1)]
        mmax = 0
        p = 0
        for i in range(len(s1)):
            for j in range(len(s2)):
                if s1[i] == s2[j]:
                    m[i+1][j+1] = m[i][j]+1
                    if m[i+1][j+1] > mmax:
                        mmax = m[i+1][j+1]
                        p = i+1
        all_rrrr[s2] = len(s1[p-mmax:p])

    all_rrrr = sorted(all_rrrr.items(), key=lambda item: item[1], reverse=True)
    return all_rrrr[: 6]  # 查找最相似的Top6答案


def evaluate_line(g, sess, seq):
    all_qwe = find_lcsubstr_spe(seq, all_text)
    all_point = []
    with g.as_default():
        model = ProposedModel()  # 加载首个答案抽取的莫i选哪个
        with sess.as_default():
            for kkk_item in all_qwe:
                kkk_item = kkk_item[0]
                kkk_item = kkk_item.split(' ')
                # print('seq:', kkk_item)
                entity_predicted, score_1 = model.predict([kkk_item])  # 对问题进行答案预测
                # print(entity_predicted)
                all_point.append([kkk_item, entity_predicted[0]])  # 形成答案列表
        # print('*' * 100)
    return all_point


def too_(points=[]):
    all_test_tuple = dict()
    # print('all_test_data_split:', all_test_data_split)
    for cur_point in points:
        json_data = query(str(cur_point))  # 基于已有的答案在Neo4j知识图谱中寻找其它答案
        entity = json_data['data']  # 节点
        relations = json_data['links']  # 关系
        for k in relations:
            target, source, value = k['target'], k['source'], k['value']  # 查找三元组，有严格的指向
            # 收集所有的元组
            if entity[source]["name"] in points:
                # 不加这个判断会出Bug
                if entity[source]["name"] not in all_test_tuple.keys():
                    all_test_tuple[entity[source]["name"]] = [entity[target]["name"]]
                else:
                    all_test_tuple[entity[source]["name"]].append(entity[target]["name"])
    return all_test_tuple


# mapping
def main(original, start_text):
    max_watch = ''
    max_keshi = ''
    best_score = 0
    """
    判断问句的范围
    """


if __name__ == "__main__":

    all_qwe = open('result1.txt', mode='a', encoding='utf-8')
    all_test_data = open('../datagenerate/proposed/processed/test_labeled_relation_data.txt', mode='r',
                         encoding='utf-8').readlines()
    for kk_allq_index, kk_allq in enumerate(all_test_data):
        if (kk_allq_index >= 0) and (kk_allq_index <= 2625):
            continue
        else:
            kk_allq = kk_allq.strip().replace('\n', '').split('\t')
            vm = kk_allq[0]
            possible_result = kk_allq[1].split(',')
            trainer_one_graph = tf.Graph()  # entity detect
            trainer_one_session = tf.Session(graph=trainer_one_graph)
            g = tf.Graph()  # entity detect
            session = tf.Session(graph=g)
            test_point = evaluate_line(trainer_one_graph, trainer_one_session, vm)
            # ##
            all_possible_entity = []
            all_possible_entity1 = []
            for k in test_point:
                k = ' '.join(k[0]).replace("'", '').replace('"', '')
                qwer = too_([k])  # 得到当前问题的所有候选答案
                if qwer is not None:
                    for kkk in qwer.values():
                        if kkk not in all_possible_entity1:
                            all_possible_entity1.append(kkk)  # 去除重复答案
                    all_possible_entity.append(qwer)
            if len(all_possible_entity1) > 0:  # 如果找到了答案
                # print('所有的可能实体：\t\t', all_possible_entity1)
                # print('概率\t')
                all_qwer = dict()
                for kkkk_index in all_possible_entity1:
                    count_all = 0
                    for aq in kkkk_index:
                        if aq in possible_result:
                            count_all += 1  # 判断答案找全率
                    all_qwer[' '.join(kkkk_index)] = count_all / len(possible_result)
                # all_qwer = sorted(all_qwer.items(), key=lambda item: item[1], reverse=True)
                # ####
                res_q = dict()
                res_q_1 = []
                qqqqqqqqqqq = []
                for kkkkkk in all_possible_entity1:
                    qqqqqqqqqqq.extend(kkkkkk)
                # print('qqqqqqqqqqq:', qqqqqqqqqqq)
                # print('possible_result:', possible_result)
                # print('*' * 100)

                for k in range(2, len(qqqqqqqqqqq), 1):
                    pq = qqqqqqqqqqq[: k]
                    cqw = 0
                    for aq in pq:
                        if aq in possible_result:
                            cqw += 1
                    if cqw not in res_q_1:
                        res_q_1.append(cqw)
                        res_q[' '.join(pq)] = min(1., cqw / len(possible_result))
                all_qwer.update(res_q)
                all_qwer = sorted(all_qwer.items(), key=lambda item: item[1], reverse=True)
                # print('结果:', all_qwer)
                qwertq = 0
                for qwe in all_qwer:
                    qwe = [str(xx) for xx in qwe]
                    all_qwe.write('\t'.join(qwe) + '\n')
                    qwertq += 1
                all_qwe.write('\n')
                all_qwe.write('++++++++++++++++Sample:{}'.format(str(kk_allq)))
                all_qwe.write('++++++++++++++++Sample ID:{}'.format(str(kk_allq_index)))
                all_qwe.write('\n')
                all_qwe.flush()
                # for
                # print(too_(['which kennedy died first']))





