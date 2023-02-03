
   [dataprocess.py] 读取原始数据集[train.json, test.json]
   [data_remanage.py] 预处理数据集，得到训练集的详细信息
   [dataread.py] 生成训练集
   [data_parse_.py] 训练过程中的数据迭代器
                    训练集和测试集分开迭代
                    输入：
                        训练集：sentence_task_train
                        测试集：sentence_task_test
                        格式:[问题，问题的首个答案，问题除首个答案之外的其它答案]
                        训练时，只用到了 [问题，问题的首个答案]
                        基于知识图谱搜索其它答案时，使用了[问题除首个答案之外的其它答案]
   