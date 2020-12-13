# -*- coding: utf-8 -*-


import pandas as pd
import numpy as np
import glob
import os
from keras.preprocessing import text, sequence
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense
from keras.callbacks import ModelCheckpoint
from sklearn.metrics import accuracy_score

import config


def load_data():
    """
        加载数据集

    """
    data_df = pd.read_csv(config.data_file, error_bad_lines=False, nrows=config.n_samples)
    raw_text_list = data_df['SentimentText'].values.tolist()
    raw_label_list = data_df['Sentiment'].values.tolist()

    # 对文本进行分词、编码等处理
    tokenizer = text.Tokenizer(num_words=config.max_n_words)
    tokenizer.fit_on_texts(raw_text_list)
    sequences = tokenizer.texts_to_sequences(raw_text_list)

    # 词汇表
    word_index = tokenizer.word_index

    # 对序列进行补长处理
    x_proc = sequence.pad_sequences(sequences, maxlen=config.max_seq_len)

    # 将标签转换为one-hot编码
    y_proc = np_utils.to_categorical(np.array(raw_label_list), num_classes=config.n_class)

    # 分割数据集
    x_train, x_test, y_train, y_test = split_data(x_proc, y_proc)

    print('训练集样本数：', x_train.shape[0])
    print('测试集样本数：', x_test.shape[0])

    return word_index, x_train, x_test, y_train, y_test


def split_data(x_proc, y_proc, test_size=0.2):
    """
        分割数据集
    """
    n_test_samples = int(test_size * x_proc.shape[0])
    x_train = x_proc[:-n_test_samples]
    y_train = y_proc[:-n_test_samples]
    x_test = x_proc[-n_test_samples:]
    y_test = y_proc[-n_test_samples:]

    return x_train, x_test, y_train, y_test


def build_model():
    """
        建立模型
    """
    model = Sequential()
    model.add(Embedding(config.max_n_words, config.embedding_nodes, input_length=config.max_seq_len))
    model.add(LSTM(config.lstm_nodes))
    model.add(Dense(config.n_class, activation='softmax'))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    print(model.summary())

    return model


def train_model(model, x_train, y_train):
    """
        训练模型
    """
    # 设置checkpoint
    filepath = os.path.join(config.output_path, '{epoch:02d}-{val_acc:.2f}.h5')
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True)
    callbacks_list = [checkpoint]
    # 模型训练过程中会将迭代中最优的训练参数进行保存
    model.fit(x_train, y_train, validation_split=0.2, epochs=config.n_epoch, batch_size=config.batch_size,
              callbacks=callbacks_list)
    return model


def do_prediction(model, x_test, y_test):
    """
        预测
    """
    # 按时间获取最新的模型参数文件
    weight_files = glob.glob(os.path.join(config.output_path, '*.h5'))
    weight_files.sort(key=os.path.getmtime)
    best_weight_file = weight_files[-1]

    # 加载模型参数
    model.load_weights(best_weight_file)

    # 预测标签
    y_pred = model.predict(x_test)

    y_test_labels = np.argmax(y_test, axis=1)
    y_pred_labels = np.argmax(y_pred, axis=1)

    print('准确率: ', accuracy_score(y_test_labels, y_pred_labels))
