import sys
import main_window
# from main_window import Ui_MainWindow
# from PyQt5 import QtWidgets
# from PyQt5.QtWidgets import QApplication, QMainWindow, QMessageBox
import random
import linecache

import numpy as np
import pickle as pk
import transformer as trans
import tensorflow as tf
import sys, os, time, codecs, pdb
import utils.tf_funcs as func
from sklearn.model_selection import KFold
from sklearn.model_selection import ParameterGrid

import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# tf.contrib.resampler
# from tensorflow.contrib.nccl.python.ops import nccl_ops
# nccl_ops._maybe_load_nccl_ops_so()


FLAGS = tf.app.flags.FLAGS


# class MainWindow(QtWidgets.QMainWindow,Ui_MainWindow):
#     def __init__(self, parent=None):
#         super(MainWindow, self).__init__(parent=parent)
#         self.setupUi(self)
#
#     def New(self):
#         senID = random.randint(1, 2105)
#         the_line = linecache.getline('./data/datacsv_2105.csv', senID)
#         the_line = the_line.strip().split(',')
#         the_line = the_line[2]
#         self.clause_edt.setPlainText(the_line)
#
#         #生成测试集
#         x_data, y_data, sen_len_data, doc_len_data, word_distance, word_embedding, pos_embedding = func.load_data()
#         te_x = x_data[senID]
#         te_y = y_data[senID]
#         te_sen_len =  sen_len_data[senID]
#         te_doc_len = doc_len_data[senID]
#         te_word_dis = word_distance[senID]
#
#         test = [te_x, te_y, te_sen_len, te_doc_len, te_word_dis, 1., 1.]
#         # print (test)
#
#         #reload训练好的模型
#         sess = tf.Session()
#
#         saver = tf.train.import_meta_graph('./run/model.ckpt.meta')
#         saver.restore(sess, tf.train.latest_checkpoint('./run'))
#
#         graph = tf.get_default_graph()
#
#         # x = graph.get_tensor_by_name("x:0")
#         # word_dis = graph.get_tensor_by_name("word_dis:0")
#         # sen_len = graph.get_tensor_by_name("sen_len:0")
#         # doc_len = graph.get_tensor_by_name("doc_len:0")
#         # keep_prob1 = graph.get_tensor_by_name("keep_prob1:0")
#         # keep_prob2 = graph.get_tensor_by_name("keep_prob2:0")
#         # y = graph.get_tensor_by_name("y:0")
#         # pred_y_op = graph.get_tensor_by_name("pred_y_op:0")
#         #
#         # placeholders = [x, word_dis, sen_len, doc_len, keep_prob1, keep_prob2, y]
#         #
#         # #将测试集传入模型进行训练
#         # with tf.Session as sess:
#         #     id = sess.run(
#         #     [pred_y_op], feed_dict=dict(zip(placeholders, test)))
#
#         #通过训练结果找出原因子句
#
#         self.cause_edt.setPlainText("原因子句")
#
#     def Ecc(self):
#         clause = self.clause_edt.toPlainText()
#         if clause == "":
#             QMessageBox.warning(self, "警告", "请输入句子")
#         else:
#             # 生成测试集
#
#             # reload训练好的模型
#
#             # 将测试集传入模型进行训练
#
#             # 通过训练结果找出原因子句
#
#             #打印原因子句
#             self.cause_edt.setPlainText(clause)
#



if __name__ == '__main__':
    # app = QApplication(sys.argv)
    # MainWindow = MainWindow()
    # MainWindow.show()
    #
    # # tf.app.run()
    # sys.exit(app.exec_())

    senID = random.randint(1, 2105)
    the_line = linecache.getline('./data/datacsv_2105.csv', senID)
    the_line = the_line.strip().split(',')
    the_line = the_line[2]
    # self.clause_edt.setPlainText(the_line)

    # 生成测试集
    x_data, y_data, sen_len_data, doc_len_data, word_distance, word_embedding, pos_embedding = func.load_data()
    te_x = x_data[senID]
    te_y = y_data[senID]
    te_sen_len = sen_len_data[senID]
    te_doc_len = doc_len_data[senID]
    te_word_dis = word_distance[senID]

    test = [te_x, te_y, te_sen_len, te_doc_len, te_word_dis, 1., 1.]
    # print (test)

    # reload训练好的模型
    sess = tf.compat.v1.Session

    saver = tf.train.import_meta_graph('./run/model.ckpt.meta', clear_devices=True)
    saver.restore(sess, tf.train.latest_checkpoint('./run'))

    graph = tf.get_default_graph()

    print("11111111")
    # x = graph.get_tensor_by_name("x:0")
    # word_dis = graph.get_tensor_by_name("word_dis:0")
    # sen_len = graph.get_tensor_by_name("sen_len:0")
    # doc_len = graph.get_tensor_by_name("doc_len:0")
    # keep_prob1 = graph.get_tensor_by_name("keep_prob1:0")
    # keep_prob2 = graph.get_tensor_by_name("keep_prob2:0")
    # y = graph.get_tensor_by_name("y:0")
    # pred_y_op = graph.get_tensor_by_name("pred_y_op:0")
    #
    # placeholders = [x, word_dis, sen_len, doc_len, keep_prob1, keep_prob2, y]
    #
    # #将测试集传入模型进行训练
    # with tf.Session as sess:
    #     id = sess.run(
    #     [pred_y_op], feed_dict=dict(zip(placeholders, test)))

    # 通过训练结果找出原因子句