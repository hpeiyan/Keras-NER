# -*- coding: utf-8 -*-
import tensorflow as tf

tf.app.flags.DEFINE_string("src_file", 'resource/source.txt', "Training data.")
tf.app.flags.DEFINE_string("tgt_file", 'resource/target.txt', "labels.")
tf.app.flags.DEFINE_string("pred_file", 'resource/predict.txt', "predict data.")
tf.app.flags.DEFINE_string("predict_label", 'resource/predict_label.txt', "predict label data.")
tf.app.flags.DEFINE_string("source_vocb_path", 'resource/source_vocb.txt', "source vocabulary.")
tf.app.flags.DEFINE_string("tgt_vocb_path", 'resource/tgt_vocb.txt', "targets.")
tf.app.flags.DEFINE_string("model_weight_path", 'model/best.model.weight', "model weight save path")

tf.app.flags.DEFINE_integer("max_features", 1000, "the max feature about input data")
tf.app.flags.DEFINE_integer("embedding_size", 30, "the size output about embedding")
tf.app.flags.DEFINE_integer("max_sequence", 100, "max sequence length.")

tf.app.flags.DEFINE_integer("batch_size", 32, "batch size.")
tf.app.flags.DEFINE_integer("epochs", 20, "epoch.")
tf.app.flags.DEFINE_float("dropout", 0.6, "drop out")

tf.app.flags.DEFINE_string("mode", 'online', "dev | online")
tf.app.flags.DEFINE_integer('max_sample', 2500, 'the max sample in dev mode')

tf.app.flags.DEFINE_string("action", 'predict', "train | predict")

FLAGS = tf.app.flags.FLAGS
