# coding=utf-8
import os
import csv
import pickle as pkl
import numpy as np
import torch
from gensim.models import word2vec
from transformers import BertTokenizer, BertModel
path = 'Testing'


def bert_read1():
    # word2id = pkl.load(open('../datagenerate/proposed/data/vocb', mode='rb'))  # 问题字典
    word2id_ans = pkl.load(open('../datagenerate/proposed/data/ans_dict.pkl', mode='rb'))  # 答案字典
    print('all num word2id_ans:', list(word2id_ans.keys()))
    print('all num in merged word2id_question:', len(list(word2id_ans.keys())))
    # ###
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased', output_hidden_states=True)  # 加载Bert预训练模型
    model.to("cpu:0")  # 本机只有CPU，有GPU的话，请修改为GPU
    model.eval()
    all_data_embedding = np.float32(np.random.uniform(-0.25, 0.25, (len(word2id_ans), 768)))  # 截断正态分布，初始化，
    for cur_char in list(word2id_ans.keys()):  # 遍历所有的分词，每一个分词直接作为Bert的输入，输出分词对应的向量
        cur_char = cur_char.strip().split(' ')[0].strip()
        tokenized_text = tokenizer.tokenize(cur_char)
        indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)  # 得到每个词在词表中的索引
        segments_ids = [1] * len(tokenized_text)  # 分词索引
        tokens_tensor = torch.tensor([indexed_tokens]).to("cpu:0")
        segments_tensors = torch.tensor([segments_ids]).to("cpu:0")

        try:
            with torch.no_grad():
                outputs = model(tokens_tensor, segments_tensors)
                hidden_states = outputs[2]

            token_embeddings = torch.stack(hidden_states, dim=0)
            token_embeddings.size()
            token_embeddings = torch.squeeze(token_embeddings, dim=1)
            token_embeddings.size()
            token_embeddings = token_embeddings.permute(1, 0, 2)  # 调换顺序
            token_embeddings.size()  # 进行3次叠加，满足Bert对输入的要求
            token_vecs_sum = [torch.sum(layer[-4:], 0) for layer in
                              token_embeddings]  # 对最后四层求和 [number_of_tokens, 768]
            print('shape of token_vecs_sum:', word2id_ans[cur_char], cur_char, token_vecs_sum[-1].shape)
            all_data_embedding[word2id_ans[cur_char]] = token_vecs_sum[-1].numpy()  # 得到768维向量，替换正太初始化向量
        except Exception as ex:
            print('Qwer')
            pass
    pkl.dump(all_data_embedding, open('ans_bert_embedding.pkl', mode='wb'))


if __name__ == "__main__":
    # create_data()
    # vec_train()
    # init_word_embedding()
    #
    # vec_train() 和 init_word_embedding() 是w2c的内容，目前不用了
    #
    # bert_read()
    bert_read1()
    pass