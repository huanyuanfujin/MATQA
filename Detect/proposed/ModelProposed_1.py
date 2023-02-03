import os
import time
import numpy as np
import pickle as pkl
import tensorflow as tf
from Detect.proposed.config import Config
from Detect.proposed.res_block import inference  # 残差模型
from Detect.tool.data_parse_ import Data_Inter  # 数据迭代程序
from tensorflow.contrib.rnn import LSTMCell  # 双向LSTM
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score  # 4准确率
seq_length = 38
# seq_length = 88


def check_multi_path(path):
    """
    创建不存在的目录
    :param path:
    :return:
    """
    assert isinstance(path, str) and len(path) > 0
    if '\\' in path:
        path.replace('\\', '/')
    childs = path.split('/')
    root = childs[0]
    for index, cur_child in enumerate(childs):
        if index > 0:
            root = os.path.join(root, cur_child)
        if not os.path.exists(root):
            os.mkdir(root)


class ProposedModel:
    def __init__(self, param_config=None,
                 model_save_path=None,
                 record_path=None,
                 label_path=None):
        self.config = Config()
        self.model_path = self.config.model_saved_path
        self.log_file_path = self.config.logging_file_saved_path
        self.record_path = record_path,
        self.label_path = label_path,
        self.update_embedding = self.config.update_embedding
        # 迭代程序所需要的文件
        self.data_inter = Data_Inter(batch_size=self.config.batch_size,
                                     task_sentence_path=['../datagenerate/proposed/data/sentence_task_train',
                                                         '../datagenerate/proposed/data/sentence_task_train'],
                                     intent2id='../datagenerate/proposed/data/task2id_id2task',
                                     vocb_path='../datagenerate/proposed/data/vocb')  # 迭代器。
        self.word2id = pkl.load(open('../datagenerate/proposed/data/vocb', mode='rb'))  # 获取本地存放的字典。
        self.tag2id = pkl.load(open('../datagenerate/proposed/data/task2id_id2task', mode='rb'))['task2id']
        self.embeddings = pkl.load(open('../bert_tokenizer_init/bert_embedding.pkl', mode='rb'))
        self.add_placeholders()
        self.build_layer_op()
        self.loss_op()
        self.trainstep_op()
        # print('self.tag2id:', self.tag2id)

    def add_placeholders(self):
        self.word_ids = tf.placeholder(tf.int32, shape=[self.config.batch_size, seq_length], name="word_ids1")
        self.task_targets_kw = tf.placeholder(tf.int64, [self.config.batch_size], name='task_targets_kw')
        self.sequence_lengths = tf.placeholder(tf.int32, shape=[None], name="sequence_lengths1")
        self.lr_pl = tf.placeholder(dtype=tf.float32, shape=[], name="lr")
        self.seta = tf.get_variable(name="W_task2",
                                    shape=[2, 2],
                                    initializer=tf.contrib.layers.xavier_initializer(),
                                    dtype=tf.float32,
                                    trainable=True)

    def bi_lstm_layer(self, hidden_num, pre_output, fw_name, bw_name, sqw=None, auto_use=False):
        """
        多层双向LSTM，由于有多层堆叠，设置参数共享
        :param hidden_num:
        :param pre_output:
        :param fw_name:
        :param bw_name:
        :param sqw:
        :param auto_use:
        :return:
        """
        with tf.variable_scope("a", reuse=tf.AUTO_REUSE):
            cell_fw1 = LSTMCell(hidden_num, name=fw_name)
            cell_bw1 = LSTMCell(hidden_num, name=bw_name)
            (output_fw_seq1, output_bw_seq1), (encoder_fw_final_state1, encoder_bw_final_state1) = \
                tf.nn.bidirectional_dynamic_rnn(
                    cell_fw=cell_fw1,
                    cell_bw=cell_bw1,
                    inputs=pre_output,
                    sequence_length=sqw,
                    dtype=tf.float32)
            output = (output_fw_seq1 + output_bw_seq1) / 2  # 每个时间步的输出
            encoder_final_state_h = tf.concat((encoder_fw_final_state1.h, encoder_bw_final_state1.h), axis=1)
            return output, encoder_final_state_h

    def bi_lstm_layer_onne_use(self, hidden_num, pre_output, fw_name, bw_name):
        with tf.variable_scope("a_none", reuse=tf.AUTO_REUSE):
            cell_fw1 = LSTMCell(hidden_num, name=fw_name)
            cell_bw1 = LSTMCell(hidden_num, name=bw_name)
            (output_fw_seq1, output_bw_seq1), (encoder_fw_final_state1, encoder_bw_final_state1) = \
                tf.nn.bidirectional_dynamic_rnn(
                    cell_fw=cell_fw1,
                    cell_bw=cell_bw1,
                    inputs=pre_output,
                    dtype=tf.float32)
            output = (output_fw_seq1 + output_bw_seq1) / 2  # 每个时间步的输出
            encoder_final_state_h = (encoder_fw_final_state1.h + encoder_bw_final_state1.h) / 2
            return output, encoder_final_state_h

    def create_variables(self, name, shape, initializer=tf.contrib.layers.xavier_initializer(), is_fc_layer=False):
        # 创建变量
        regularizer = tf.contrib.layers.l2_regularizer(scale=0.0002)
        new_variables = tf.get_variable(name, shape=shape, initializer=initializer,
                                        regularizer=regularizer)
        return new_variables

    def build_layer_op(self):
        with tf.variable_scope("words"):
            _word_embeddings = tf.Variable(self.embeddings,
                                           dtype=tf.float32,
                                           trainable=self.update_embedding,
                                           name="_word_embeddings")
            word_embeddings = tf.nn.embedding_lookup(params=_word_embeddings,
                                                     ids=self.word_ids,
                                                     name="word_embeddings")

        self.word_embeddings = word_embeddings
        print('self.word_embeddings:', self.word_embeddings)
        print('self.word_embeddings::::::::::', self.word_embeddings)
        word_embeddings_tmp = tf.expand_dims(self.word_embeddings, axis=3)
        # feature_embedded_resnet = tf.squeeze(inference(word_embeddings_tmp, 1, reuse=False), axis=3)
        feature_embedded_resnet = tf.layers.flatten(inference(word_embeddings_tmp, 1, reuse=False))  # 残差网络提取的特征
        print('feature_embedded_input:', feature_embedded_resnet)
        finall_feature_cause, encoder_final_state_h = self.bi_lstm_layer(hidden_num=100,
                                                                         pre_output=self.word_embeddings,
                                                                         fw_name='cell_fw11', bw_name='cell_bw11',
                                                                         # sqw=self.sequence_lengths
                                                                         )
        print('finall_feature1:', encoder_final_state_h)
        # 残差网络提取的特征和多层LSTM提取的特征进行拼接，设置了2个贡献度因子
        feature_merged = tf.concat([encoder_final_state_h * self.seta[0][0], feature_embedded_resnet * self.seta[1][0]], axis=1)
        feature_merged_intent = encoder_final_state_h  # intend feature
        print('shape of all:feature_merged:{}\t{}'.format(feature_merged.shape, feature_merged_intent.shape))
        with tf.variable_scope("proj"):
            w_task1 = tf.get_variable(name="W_task1",
                                      shape=[feature_merged.shape[1], len(self.tag2id)],  # classification
                                      initializer=tf.contrib.layers.xavier_initializer(),
                                      dtype=tf.float32)
            b_task1 = tf.get_variable(name="b_task1",
                                      shape=[len(self.tag2id)],
                                      initializer=tf.zeros_initializer(),
                                      dtype=tf.float32)

            print('finall_feature1:', feature_merged)
            slot_logits_kw = tf.add(tf.matmul(feature_merged, w_task1), b_task1)
            print('距离损失:', slot_logits_kw)
            # 损失1
            self.softmax_score_kw = tf.nn.softmax(slot_logits_kw)
            self.task = tf.argmax(slot_logits_kw, axis=1)  # 全连接，对第一个答案的预测
            # self.task1 = slot_logits_kw
            # #########################################
            self.acc_slot = tf.reduce_mean(tf.cast(tf.equal(self.task, self.task_targets_kw), dtype=tf.float32))
            cross_entropy_kw = tf.nn.softmax_cross_entropy_with_logits(
                labels=tf.one_hot(self.task_targets_kw, depth=len(self.tag2id), dtype=tf.float32),
                logits=slot_logits_kw)
            #
            print('cross_entropy_kw:', cross_entropy_kw)
            self.loss_task_kw = tf.reduce_mean(cross_entropy_kw)
            self.loss_task = self.loss_task_kw
            self.all_loss = self.loss_task
            self.acc = self.acc_slot

    def loss_op(self):
        self.loss = self.all_loss  # 任务识别的损失

    def trainstep_op(self):
        """
        训练节点.
        """
        with tf.variable_scope("train_step"):
            self.global_step = tf.Variable(0, name="global_step", trainable=False)  # 全局训批次的变量，不可训练。
            if self.config.optimizer == 'Adam':
                optim = tf.train.AdamOptimizer(learning_rate=.0001)
            elif self.config.optimizer == 'Adadelta':
                optim = tf.train.AdadeltaOptimizer(learning_rate=.0001)
            elif self.config.optimizer == 'Adagrad':
                optim = tf.train.AdagradOptimizer(learning_rate=.0001)
            elif self.config.optimizer == 'RMSProp':
                optim = tf.train.RMSPropOptimizer(learning_rate=.0001)
            elif self.config.optimizer == 'Momentum':
                optim = tf.train.MomentumOptimizer(learning_rate=.0001, momentum=0.9)
            elif self.config.optimizer == 'SGD':
                optim = tf.train.GradientDescentOptimizer(learning_rate=.0001)
            else:
                optim = tf.train.GradientDescentOptimizer(learning_rate=.0001)
            grads_and_vars = optim.compute_gradients(self.loss)
            grads_and_vars_clip = [[tf.clip_by_value(g, -self.config.get_clip, self.config.get_clip), v] for g, v in
                                   grads_and_vars]
            self.train_op = optim.apply_gradients(grads_and_vars_clip, global_step=self.global_step)

    def get_word_em_bf(self, index_input):
        # 使用bert初始化一个句子
        new_data = np.zeros(shape=[32, 21, 80, 128])
        for cur_batch in range(32):
            for cur_repos in range(21):
                for index, cur_word in enumerate(index_input[cur_batch][cur_repos]):
                    new_data[cur_batch][cur_repos][index] = self.embeddings[cur_word]
        return new_data

    def pad_sequences(self, sequences, pad_mark=0, predict=False):
        """
        批量的embedding，其中rowtext embedding的长度要与slots embedding的长度一致，不然使用crf时会出错。
        :param sequences: 批量的文本格式[[], [], ......, []]，其中子项[]里面是一个完整句子的embedding（索引。）
        :param pad_mark:  长度不够时，使用何种方式进行padding
        :param predict:  是否是测试
        :return:
        """

        max_len = seq_length
        seq_list, seq_len_list = [], []
        for seq in sequences:
            seq = list(seq)
            if predict:
                seq = list(map(lambda x: self.word2id.get(x, 0), seq))
            seq_ = seq[: max_len] if len(seq) >= max_len else seq + [pad_mark] * (max_len - len(seq))  # 求得最大的索引长度。
            seq_list.append(seq_)
            seq_len_list.append(min(len(seq), max_len))
        return seq_list, seq_len_list

    def batch_evaluate(self, predict__, real__, stride__):
        """
        批量计算准确率，这个准确率计算的更加准确，避免错位，
        本项目中没用到，对准确率要求严格的项目中有用到。
        :param predict__:
        :param real__:
        :param stride__:
        :return:
        """
        acc = []
        f1 = []
        pre = []
        recall = []
        k_batch = len(predict__) // stride__
        for j in range(0, k_batch, 1):
            acc_tmp = accuracy_score(predict__[j * stride__: (j + 1) * stride__],
                                     real__[j * stride__: (j + 1) * stride__])
            f1_tmp = f1_score(predict__[j * stride__: (j + 1) * stride__], real__[j * stride__: (j + 1) * stride__],
                              average='weighted')
            pre_tmp = precision_score(predict__[j * stride__: (j + 1) * stride__],
                                      real__[j * stride__: (j + 1) * stride__], average='weighted')
            recall_tmp = recall_score(predict__[j * stride__: (j + 1) * stride__],
                                      real__[j * stride__: (j + 1) * stride__], average='weighted')
            acc.append(acc_tmp)
            f1.append(f1_tmp)
            pre.append(pre_tmp)
            recall.append(recall_tmp)

        # ####
        acc = sum(acc) / len(acc)
        f1 = sum(f1) / len(f1)
        pre = sum(pre) / len(pre)
        r = sum(recall) / len(recall)
        return acc, f1, pre, r

    def train(self, log_file=None):
        """
            数据由一个外部迭代器提供。
        """
        if log_file is None:
            log_file = open(self.config.logging_file_saved_path.__add__('proposed_mt.txt'), mode='w', encoding='utf-8')
            print('Not lazy......')
        saver = tf.train.Saver(tf.global_variables())
        with tf.Session() as sess:
            # sess.run(tf.global_variables_initializer())
            ckpt_file = tf.train.latest_checkpoint('../model_save'.__add__('/'))
            if ckpt_file is not None:
                print('ckpt_file:', ckpt_file)
                saver.restore(sess, ckpt_file)
            else:
                sess.run(tf.global_variables_initializer())
            batches_recording = 0
            for epoch_index in range(0, 300000, 1):
                start_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
                sentence_, tasks_all, sentence_length = self.data_inter.next(seq_length)  #
                print('sentence_:', sentence_.shape, tasks_all)
                sentence_train, _ = self.pad_sequences(sentence_)
                _, loss_train, acc_cur, step_num_, qq = sess.run([self.train_op, self.loss, self.acc,
                                                                  self.global_step, self.task], feed_dict={
                    self.lr_pl: 0.001,
                    self.word_ids: sentence_train,
                    self.task_targets_kw: list(tasks_all),
                    self.sequence_lengths: list(sentence_length)
                })

                if epoch_index % 1 == 0:
                        sentence_test, task_test, sentence_test_length = \
                            self.data_inter.next_test(seq_length)  # 迭代器，每次取出一个batch块.

                        sentence_test, _ = self.pad_sequences(sentence_test)
                        print('sentence_test_length:', sentence_test_length)
                        print('sentence_test_length:', np.array(sentence_test).shape)
                        loss_test, acc_kw_test, step_num_, task_kw_for_score = sess.run(
                            [self.loss, self.acc,
                             self.global_step, self.task], feed_dict={
                                self.lr_pl: 0.001,
                                self.word_ids: sentence_test,
                                self.task_targets_kw: list(task_test),
                                self.sequence_lengths: list(sentence_test_length)
                            })
                        # 加入其它的准确率
                        # #####################################
                        f1 = f1_score(task_test, task_kw_for_score, average='macro')
                        pre = precision_score(task_test, task_kw_for_score, average='macro')
                        r = recall_score(task_test, task_kw_for_score, average='macro')
                        # ####################################
                        if log_file is not None:
                            log_file.write('time:'.__add__(start_time).__add__('\tepoch: ').
                                           __add__(str(epoch_index + 1)).__add__('\tstep:').
                                           __add__(str(batches_recording + epoch_index)).
                                           __add__('\tloss:{:.4}').
                                           __add__('\tacc:{:.4}').
                                           __add__('\tf1:{:.4}').
                                           __add__('\tpre:{:.4}').
                                           __add__('\trecall:{:.4}').
                                           __add__('\n').format(loss_test, acc_kw_test, f1, pre, r))
                            log_file.flush()
                        print('time:'.__add__(start_time).__add__('\tepoch: ').
                              __add__(str(epoch_index + 1)).__add__('\tstep:').
                              __add__(str(batches_recording + epoch_index)).
                              __add__('\tloss:{:.4}').
                              __add__('\tacc:{:.4}').
                              __add__('\tf1:{:.4}').
                              __add__('\tpre:{:.4}').
                              __add__('\trecall:{:.4}').
                              __add__('\n').format(loss_test, acc_kw_test, f1, pre, r))
                if epoch_index % 2000 == 0:
                    check_multi_path(self.model_path)
                    saver.save(sess, self.model_path, global_step=epoch_index)
                # except Exception as ex:
                #     print('batch error......')
        if log_file is not None:
            log_file.close()

    def get_feed_dict(self, seqs, labels=None, lr=None, dropout=None, tag=None, predicted=False):
        """

        :param seqs:  训练的batch块
        :param labels:  实体标签
        :param lr:  学利率
        :param dropout:  活跃的节点数，全连接层
        :return: feed_dict  训练数据
        :return: predicted  测试标志
        """
        # print('seqs:', seqs.shape)
        word_ids, seq_len_list = self.pad_sequences(seqs, pad_mark=0, predict=predicted)
        # print('word_ids:', np.array(word_ids).shape)
        # print('seq_len_list:', seq_len_list)
        feed_dict = {self.word_ids: word_ids,  # embedding到同一长度
                     self.sequence_lengths: seq_len_list,  # 实际长度。
                     }
        if labels is not None:
            feed_dict[self.task_targets_kw] = labels[0]
        if lr is not None:
            feed_dict[self.lr_pl] = lr
        return feed_dict, seq_len_list

    def predict(self, seq):
        """

        :param sess:
        :param seqs:
        :param predicted:
        :return: label_list
                 seq_len_list
        """
        saver = tf.train.Saver(tf.global_variables())
        with tf.Session() as sess:
            ckpt_file = tf.train.latest_checkpoint('../model_save'.__add__('/'))
            saver.restore(sess, ckpt_file)
            sentence_, _ = self.pad_sequences(seq)
            word_ids, seq_len_list = self.pad_sequences(sentence_, pad_mark=0, predict=True)
            task_result, soft_score = sess.run([self.task, self.softmax_score_kw],
                                feed_dict={
                                    self.lr_pl: 0.001,
                                    self.word_ids: word_ids,
                                    self.sequence_lengths: [len(seq[0])]
                                })
            return task_result, soft_score[0][task_result[0]]


if __name__ == "__main__":
    test_model = ProposedModel()
    test_model.train()
