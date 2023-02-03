import pickle as pkl


class Config:
    def __init__(self):
        """
        配置文件
        """
        self.seta = 1.1
        self.beta = 1.5
        self.epoch = 200
        self.get_clip = 10.
        self.batch_size = 32
        # self.batch_size = 1
        self.repost = 21
        self.learning_rate = 0.001
        self.keep_dropout = .5
        self.optimizer = 'Adam'
        self.shuffle = True
        self.sequence_length = 80
        self.embedding_size = 128
        self.get_embedding_dim = 128
        self.iter_routing = 3  # 胶囊网络的路由层数
        self.update_embedding = True  # 训练的时候更新映射
        self.hidden_units = 64
        self.model_saved_path = '../model_save/'
        self.logging_file_saved_path = '../logs/'
        #


class DataProcessConfig:
    def __init__(self):
        self.content_length = 21

    @property
    def content_len(self):
        return self.content_length
