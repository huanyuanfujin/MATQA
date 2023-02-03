import csv
# 基于neo4j/create_neo4j_data.py生成的roles.csv生成TransE所需要的输入
data_csv = csv.reader(open('../csv/roles.csv', encoding='utf-8', mode='r'))
en2id = open('en2id.txt', mode='w', encoding='utf-8')
re2id = open('re2id.txt', mode='w', encoding='utf-8')
train = open('train.txt', mode='w', encoding='utf-8')
print('1')
all_relation = dict()
all_id_2_relation = dict()
all_entity = dict()
for j in data_csv:
    # print(j)
    s, o, l = j[:]
    if l not in all_relation.keys():
        all_relation[l] = len(all_relation)
        all_id_2_relation[len(all_relation) - 1] = 're' + str(len(all_relation) - 1)
    if s not in all_entity.keys():
        all_entity[s] = len(all_entity)
    if o not in all_entity.keys():
        all_entity[o] = len(all_entity)
for jj_k, jj_value in zip(all_entity.keys(), all_entity.values()):
    en2id.write(str(jj_k) + "\t" + str(jj_value) + '\n')

for jjj_k, jjj_value in zip(all_relation.keys(), all_relation.values()):
    re2id.write('re' + str(jjj_value) + "\t" + str(jjj_value) + '\n')

data_csv = csv.reader(open('../csv/roles.csv', encoding='utf-8', mode='r'))
print(all_id_2_relation)
for jk in data_csv:
    print('uyt')
    train.write(jk[0] + '\t' + jk[1] + '\t' + str(all_id_2_relation[all_relation[jk[2]]]) + '\n')