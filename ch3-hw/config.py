# -*- coding: utf-8 -*-


import os

# 数据集路径
data_file = '/wenxin/training.1600000.processed.noemoticon.csv'
# 结果保存路径
output_path = './output'
if not os.path.exists(output_path):
    os.makedirs(output_path)

# 迭代次数
n_epoch = 10

# 批大小
batch_size = 128

# embdding层结点数
embedding_nodes = 128

# LSTM结点数s
lstm_nodes = 256

# 词汇表大小
max_n_words = 20000

# 序列最大长度
max_seq_len = 100

# 样本数量
n_samples = 100000

# 类别个数
n_class = 2

# 是否加载训练好的模型参数
load_model = True
