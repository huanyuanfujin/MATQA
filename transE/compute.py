# coding=utf-8
import numpy as np
import codecs
import operator
import json
import csv
import pickle as pkl
from transE import data_loader


def dataloader(entity_file, relation_file):
    # entity_file: entity \t embedding
    entity_dict = {}
    relation_dict = {}

    with codecs.open(entity_file) as e_f:
        lines = e_f.readlines()
        for line in lines:
            entity,embedding = line.strip().split('\t')
            embedding = json.loads(embedding)
            entity_dict[entity] = embedding

    with codecs.open(relation_file) as r_f:
        lines = r_f.readlines()
        for line in lines:
            relation,embedding = line.strip().split('\t')
            embedding = json.loads(embedding)
            relation_dict[relation] = embedding

    return entity_dict,relation_dict


def distance(h, r, t):
    h = np.array(h)
    r=np.array(r)
    t = np.array(t)
    s=h+r-t
    print('1')
    return np.linalg.norm(s)


def read_entity():
    entity = csv.reader(open('Source/csv/entity.csv', mode='r', encoding='utf-8'))
    entity_dit = dict()
    skip_title = True
    for q in entity:
        if skip_title:
            skip_title = False
            continue
        else:
            if 'en' in q[0]:
                entity_dit[q[1]] = q[0]
    return entity_dit


def get_entity(ent_1):
    entity_dit = read_entity()
    data = open('Source/reSource/en2id.txt', mode='r', encoding='utf-8').readlines()
    data_dict = dict()
    for j in data:
        j = j.strip().split('\t')
        if len(j) == 2:
            if 'en' in j[0] and ':en' not in j[0]:
                data_dict[j[0]] = ent_1[j[1]]
    #
    for k, v in zip(entity_dit.keys(), entity_dit.values()):
        entity_dit[k] = data_dict[v]
    return data_dict


if __name__ == '__main__':
    # _, _, train_triple = data_loader("")

    entity_dict, relation_dict = dataloader("entity_50dim_batch4001", "relation50dim_batch4001")
    entity_2_50_dim_vector = get_entity(entity_dict)
    pkl.dump(entity_2_50_dim_vector, open('entity_2_50_dim_vector', mode='wb'))
    print()

