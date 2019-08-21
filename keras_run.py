#!/usr/bin/env python
# encoding: utf-8
'''
@author: Ben
@license: (C) Copyright 2013-2017, Node Supply Chain Manager Corporation Limited.
@contact: 2017hpy@gmail.com
@file: keras_run.py
@time: 2019/8/15 09:42
@desc:
'''

from model import NerModel
from utils import *

if __name__ == '__main__':
    log.i('Start main function.')

    model = NerModel()
    model.train() if is_train() else model.predict()

    log.i('Process finish')
