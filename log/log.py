#!/usr/bin/env python
# encoding: utf-8
'''
@author: Ben
@license: (C) Copyright 2013-2017, Node Supply Chain Manager Corporation Limited.
@contact: 2017hpy@gmail.com
@file: log.py
@time: 2019/8/8 16:35
@desc:
'''
import logging.config

config = {
    'version': 1,
    'formatters': {
        'simple': {
            'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        },
        # 其他的 formatter
    },
    'handlers': {
        'console': {
            'class': 'logging.StreamHandler',
            'level': 'INFO',
            'formatter': 'simple'
        },
        'file': {
            'class': 'logging.FileHandler',
            'filename': './log/logging.log',
            'level': 'INFO',
            'formatter': 'simple'
        },
        # 其他的 handler
    },
    'loggers': {
        'StreamLogger': {
            'handlers': ['console'],
            'level': 'INFO',
        },
        'FileLogger': {
            # 既有 console Handler，还有 file Handler
            # 'handlers': ['console', 'file'],
            'handlers': ['file'],
            'level': 'INFO',
        },
        # 其他的 Logger
    }
}


class Log():
    def __init__(self):
        logging.config.dictConfig(config)
        self.fileLogger = logging.getLogger('FileLogger')

    @staticmethod
    def initialize():
        logging.config.dictConfig(config)

    def d(self, msg):
        self.fileLogger.debug(msg)

    def e(self, msg):
        self.fileLogger.error(msg)

    def w(self, msg):
        self.fileLogger.warning(msg)

    def i(self, msg):
        self.fileLogger.info(str(msg))
