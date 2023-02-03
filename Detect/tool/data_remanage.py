# coding=utf-8
import pickle as pkl


def get_data(data_path, config_path, word_save_path=None):
    data_q = pkl.load(open(data_path, mode='rb'))  # 加载 Detect/tool/dataprocess.py处理原始数据的结果:问题，答案二元组
    data_q_config = open(config_path, mode='w', encoding='utf-8')  # 另存
    unique_label = dict()  # 非重复答案字典
    min_ans = 1e19
    max_ans = 0
    min_question = 1e19
    max_question = 0
    ans_max = []
    for k in data_q:
        k_ans = k[1]
        print('k_ans:', k_ans)
        min_ans = min(min_ans, len(k_ans))
        # if max_ans < len(k_ans):
        #     ans_max = k_ans
        ans_max.append(len(k_ans))
        max_ans = max(max_ans, len(k_ans))
        min_question = min(min_question, len(k[0].strip().split(' ')))
        max_question = max(max_question, len(k[0].strip().split(' ')))
        for kk in k_ans:
            # kk = kk.lower()
            if kk not in unique_label:
                unique_label[kk] = len(unique_label)
    if word_save_path is not None:
        pkl.dump(unique_label, open(word_save_path, mode='wb'))  # 保存非重复问题答案字典
    print('all unique entity answer:', len(list(unique_label)))  # 5638
    ans_max = sorted(ans_max)
    ans_max_1 = list(filter(lambda x: x > 1, ans_max))

    data_q_config.write('min_ans:{}'.format(str(min_ans)) + '\n')
    data_q_config.write('max_ans:{}'.format(str(max_ans)) + ' ' + str(ans_max_1[-2]) + '\n')
    data_q_config.write('middle_more_than_1_ans:{}'.format(str(ans_max_1[len(ans_max_1) // 2])) + '\n')
    data_q_config.write('more_than_1_ans:{}'.format(str(len(ans_max_1))) + '\n')

    data_q_config.write('min_question:{}'.format(str(min_question)) + '\n')
    data_q_config.write('max_question:{}'.format(str(max_question)) + '\n')
    data_q_config.write('words:{}'.format(len(list(unique_label))) + '\n')

    print('ans_max:', len(ans_max))
    print('ans_max:', len(ans_max_1))
    print('ans_max:', ans_max_1[len(ans_max_1) // 2])
    print('ans_max:', ans_max_1)


if __name__ == '__main__':
    """
    cfg.txt
    以下配置项目的值，是对训练集深度处理之后形成的，感兴趣的话，可以自己处理下
        # 训练集的配置项
        min_ans:1----最短的答案（只有1个答案）
        max_ans:233 94-----------训练集/测试集最长的答案
        middle_more_than_1_ans:3  # 答案多余1个的问题答案排序之后答案中位数
        more_than_1_ans:1513  # 多余1个答案的问题个数
        min_question:1
        max_question:37  # 最长的问题分词
        words:3961  # 问题分词个数
    """
    get_data('../../processed/train.pkl', config_path='../../processed/cfg.txt', word_save_path='../../processed/ans_word.pkl')