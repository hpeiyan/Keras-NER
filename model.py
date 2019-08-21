#!/usr/bin/env python
# encoding: utf-8
'''
@author: Ben
@license: (C) Copyright 2013-2017, Node Supply Chain Manager Corporation Limited.
@contact: 2017hpy@gmail.com
@file: model.py
@time: 2019/8/21 17:40
@desc:
'''

from keras import Sequential
from keras import layers
from keras.layers import Bidirectional
from keras_contrib.layers import CRF
from keras.callbacks import EarlyStopping
import os
import ast
from utils import *


class NerModel():
    def __init__(self):
        self.model = self.__build_model__()

    def __build_model__(self):
        model = Sequential()
        emb = layers.Embedding(input_dim=MAX_FEATURE, output_dim=embedding_size, input_length=MAX_LEN)
        bdr = Bidirectional(layers.LSTM(64, return_sequences=True))
        crf = CRF(get_label_counts() + 1, sparse_target=True)

        model.add(emb)
        model.add(bdr)
        model.add(crf)

        model.summary()
        model.compile(optimizer='adam', loss=crf.loss_function, metrics=[crf.accuracy])
        return model

    def train(self):
        X_train, X_test, y_train, y_test = generate_data()
        elsp = EarlyStopping(monitor='val_loss', patience=3, mode='min', verbose=0)
        h = self.model.fit(X_train, y_train, batch_size=BATCH_SIZE, epochs=EPOCHS, validation_split=0.2,
                           callbacks=[elsp])
        self.model.save_weights(WEIGHT_PATH, overwrite=True)
        plot_history(h)
        test = self.model.evaluate(X_test, y_test, batch_size=BATCH_SIZE)
        log.i('Test data loss: {}, accuracy: {}'.format(test[0], test[1]))

    def predict(self):
        if not os.path.exists(WEIGHT_PATH):
            log.w('WEIGHT_PATH no exists.')
            return

        with open(pred_file, 'r') as f:
            sts_prd = f.readlines()

        with open(src_file, 'r') as f:
            sentences = f.readlines()
        tk = Tokenizer(num_words=MAX_FEATURE, filters=' \n')
        tk.fit_on_texts(sentences)

        sen_sqc = tk.texts_to_sequences(sts_prd)
        sen_sqc_pad = pad_sequences(sen_sqc, maxlen=MAX_LEN)

        model = self.model
        model.load_weights(WEIGHT_PATH)
        pre = model.predict(sen_sqc_pad, batch_size=BATCH_SIZE)

        with open(tgt_vocb_path, 'r') as f:
            s = f.read()
        tgt_dict = ast.literal_eval(s)
        tgt_dict[0] = 'padding'
        b_s = []
        for s_b in pre:
            indexs = s_b.argmax(axis=1)
            b_s.append(list(filter(lambda c: c is not 'padding', [tgt_dict.get(i) for i in indexs])))
        log.i(b_s)

        sss = []
        for ss in sen_sqc_pad:
            sss.append([tk.index_word.get(s) for s in list(filter(lambda s: s != 0, ss))])
        log.i(sss)

        log.i(len(sss[0]))
        log.i(len(str(b_s[0]).split()))

        s_a = []
        for i, s in enumerate(sss):
            s_t = b_s[i]
            w_a = []
            for i, w in enumerate(s):
                w_t = s_t[i]
                w_a.append('{}({})'.format(w, w_t))
            s_a.append(w_a)

        with open(predict_label_path, 'w') as f:
            for s in s_a:
                f.write(str(s) + '\n\n')
                log.i(s)
