#!/usr/bin/env python
# encoding: utf-8
'''
@author: Ben
@license: (C) Copyright 2013-2017, Node Supply Chain Manager Corporation Limited.
@contact: 2017hpy@gmail.com
@file: utils.py
@time: 2019/8/21 17:38
@desc:
'''

import config
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from log.log import Log
import matplotlib.pyplot as plt

log = Log()

src_file = config.FLAGS.src_file
tgt_file = config.FLAGS.tgt_file
pred_file = config.FLAGS.pred_file
MAX_SAMPLE = config.FLAGS.max_sample
EPOCHS = config.FLAGS.epochs
BATCH_SIZE = config.FLAGS.batch_size
embedding_size = config.FLAGS.embedding_size
WEIGHT_PATH = config.FLAGS.model_weight_path
mode = config.FLAGS.mode
source_vocb_path = config.FLAGS.source_vocb_path
tgt_vocb_path = config.FLAGS.tgt_vocb_path
MAX_LEN = config.FLAGS.max_sequence
MAX_FEATURE = config.FLAGS.max_features
action = config.FLAGS.action
predict_label_path = config.FLAGS.predict_label


def is_train():
    if action is 'train':
        return True
    return False


def is_dev():
    if mode is 'dev':
        return True
    return False


def get_feature_length():
    '''
    获取语料库中，所有的不同的词语数量，和最长的句子数量
    :return:
    '''
    with open(src_file, 'r') as f:
        sentences = f.readlines()
    tk = Tokenizer(filters='')
    tk.fit_on_texts(sentences)
    max_length = max([len(s.split()) for s in sentences])
    feature_counts = len(tk.word_counts)
    return feature_counts, max_length


MAX_FEATURE, _ = get_feature_length()


def get_label_counts():
    '''
    获取所有的标签数量
    :return:
    '''
    with open(tgt_file, 'r') as f:
        tgt = f.readlines()

    tk_tgt = Tokenizer(num_words=MAX_FEATURE, filters=' \n')
    tk_tgt.fit_on_texts(tgt)
    return len(tk_tgt.word_counts)


def generate_data():
    with open(src_file, 'r') as f:
        sentences = f.readlines()

    tk = Tokenizer(num_words=MAX_FEATURE, filters=' \n')
    tk.fit_on_texts(sentences)
    sen_sqc = tk.texts_to_sequences(sentences)

    sen_sqc_pad = pad_sequences(sen_sqc, maxlen=MAX_LEN)

    with open(source_vocb_path, 'w') as f:
        f.write(str(tk.index_word))

    with open(tgt_file, 'r') as f:
        tgt = f.readlines()

    tk_tgt = Tokenizer(num_words=MAX_FEATURE, filters=' \n')
    tk_tgt.fit_on_texts(tgt)
    tgt_sqc = tk_tgt.texts_to_sequences(tgt)
    tgt_sqc_pad = pad_sequences(tgt_sqc, maxlen=MAX_LEN)
    with open(tgt_vocb_path, 'w') as f:
        f.write(str(tk_tgt.index_word))

    if is_dev():
        sen_sqc_pad = sen_sqc_pad[:MAX_SAMPLE]
        tgt_sqc_pad = tgt_sqc_pad[:MAX_SAMPLE]

    log.i('sen shape: {}, tgt shape:{}'.format(sen_sqc_pad.shape, tgt_sqc_pad.shape))

    # tgt_sqc_pad = np.expand_dims(tgt_sqc_pad, 2)
    tgt_sqc_pad = tgt_sqc_pad.reshape((tgt_sqc_pad.shape[0], tgt_sqc_pad.shape[1], 1))
    X_train, X_test, y_train, y_test = train_test_split(sen_sqc_pad, tgt_sqc_pad, test_size=0.2)
    log.i('generate data finish')

    return X_train, X_test, y_train, y_test


def plot_history(h):
    acc = h.history['crf_viterbi_accuracy']
    loss = h.history['loss']
    val_acc = h.history['val_crf_viterbi_accuracy']
    val_loss = h.history['val_loss']
    epochs = range(1, len(acc) + 1)
    plt.plot(epochs, acc, label='train_acc')
    plt.plot(epochs, val_acc, label='val_acc')
    plt.legend()
    plt.title('training acc & val acc')
    plt.figure()
    plt.plot(epochs, loss, label='train_loss')
    plt.plot(epochs, val_loss, label='val_loss')
    plt.legend()
    plt.title('training loss & val loss')
    plt.show()
