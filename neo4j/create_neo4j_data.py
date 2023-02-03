# coding=utf-8
import csv
import pickle as pkl
#
# 制作知识图谱
# path = '../triplets/proposed/train.txt'
path = '../Detect/datagenerate/proposed/processed/train_labeled_relation_data.txt'
all_valid_entity = dict()  # 保存所有不重复的实体


def create_kg():  # 制作图谱
    basic_graph = open(path, mode='r', encoding='utf-8').readlines()
    all_s = []

    def create_en():
        """
        en_s, en_o, en_all
        :return:
        """
        csv_wirter = csv.writer(open('data/entity.csv', encoding='utf-8', mode='w', newline=''))
        csv_wirter.writerow(['entity:ID', 'name', ':LABEL'])
        index_all = 0
        # basic
        for cur_j_index, cur_j in enumerate(basic_graph):  # 遍历所有的主体+客体
            cur_j_q, cur_j_ans = cur_j.strip().split('\t')[:]
            cur_j_ans = cur_j_ans.strip().split(',')
            cur_j_ans.append(cur_j_q)
            for cur_index_, cur_ in enumerate(cur_j_ans):
                if cur_ not in all_s:
                    cur_ = cur_.replace("'", '').replace('"', '')
                    all_s.append(cur_)
                    index_all += 1
                    if cur_index_ == len(cur_j_ans) - 1:
                        # csv_wirter.writerow(['en' + str(index_all), cur_, 'q'])  # 颜色
                        csv_wirter.writerow(['en' + str(index_all), cur_, 's'])  # 颜色
                    else:
                        csv_wirter.writerow(['en' + str(index_all), cur_, 's'])  # 颜色
                    all_valid_entity[cur_] = 'en' + str(index_all)
                else:
                    pass

    def create_relation():
        """
        :return:
        """
        csv_wirter1 = csv.writer(open('data/roles.csv', encoding='utf-8', mode='w', newline=''))
        csv_wirter1.writerow([':START_ID', ':END_ID', ':TYPE'])  # 表头
        # basic
        print('basic_graph:', basic_graph[0])
        for cur_s_cur_r_cur_o in basic_graph:
            cur_s_cur_r_cur_o1,  cur_s_cur_r_cur_o2 = cur_s_cur_r_cur_o.strip().split('\t')
            cur_s_cur_r_cur_o1 = cur_s_cur_r_cur_o1.replace("'", '').replace('"', '')
            cur_s_cur_r_cur_o2 = cur_s_cur_r_cur_o2.strip().split(',')
            for cur_ob in cur_s_cur_r_cur_o2:
                cur_ob = cur_ob.replace("'", '').replace('"', '')
                csv_wirter1.writerow([all_valid_entity[cur_s_cur_r_cur_o1], all_valid_entity[cur_ob], 'answer'])
    create_en()
    create_relation()


if __name__ == '__main__':
    create_kg()