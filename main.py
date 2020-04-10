import sys
import main_window
from main_window import Ui_MainWindow
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QApplication, QMainWindow, QMessageBox
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

FLAGS = tf.app.flags.FLAGS


class MainWindow(QtWidgets.QMainWindow,Ui_MainWindow):
    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent=parent)
        self.setupUi(self)

    def New(self):
        senID = random.randint(1, 2105)
        the_line = linecache.getline('./data/datacsv_2105.csv', senID)
        the_line = the_line.strip().split(',')
        the_line = the_line[2]

        self.clause_edt.setPlainText(the_line)

        # 生成测试集
        x_data, y_data, sen_len_data, doc_len_data, word_distance, word_embedding, pos_embedding = func.load_data()

        print("senID:{}\n".format(senID))

        te_x = x_data[senID]
        te_x = te_x[np.newaxis, :]
        te_y = y_data[senID]
        te_y = te_y[np.newaxis, :]
        te_sen_len = sen_len_data[senID]
        te_sen_len = te_sen_len[np.newaxis, :]
        te_doc_len = np.array([doc_len_data[senID]])
        te_word_dis = word_distance[senID]
        te_word_dis = te_word_dis[np.newaxis, :]

        # print("te_x.shape:{}\n".format(te_x.shape))
        # print("te_y.shape:{}\n".format(te_y.shape))
        # print("te_sen_len.shape:{}\n".format(te_sen_len.shape))
        # print("te_doc_len:{}\n".format(te_doc_len))
        # print("te_doc_len.shape:{}\n".format(te_doc_len.shape))
        # print("te_word_dis.shape:{}\n".format(te_word_dis.shape))

        test = [te_x, te_y, te_sen_len, te_doc_len, te_word_dis, 1., 1.]
        # print (test)
        tf.reset_default_graph()

        with tf.Session() as sess:
            # reload训练好的模型

            saver = tf.train.import_meta_graph('run_final/model.ckpt.meta')
            model_file = tf.train.latest_checkpoint('run_final/')
            saver.restore(sess, model_file)

            graph = tf.get_default_graph()

            x = graph.get_tensor_by_name("x:0")
            word_dis = graph.get_tensor_by_name("word_dis:0")
            sen_len = graph.get_tensor_by_name("sen_len:0")
            doc_len = graph.get_tensor_by_name("doc_len:0")
            keep_prob1 = graph.get_tensor_by_name("keep_prob1:0")
            keep_prob2 = graph.get_tensor_by_name("keep_prob2:0")
            y = graph.get_tensor_by_name("y:0")
            pred = graph.get_tensor_by_name("pred_3:0")
            true_y_op = graph.get_tensor_by_name("true_y_op:0")
            pred_y_op = graph.get_tensor_by_name("pred_y_op:0")

            placeholders = [x, y, sen_len, doc_len, word_dis, keep_prob1, keep_prob2]

            # 将测试集传入模型进行训练
            pred_y,true_y, pred = sess.run([pred_y_op, true_y_op, pred], feed_dict=dict(zip(placeholders, test)))
            print(pred_y.shape)
            pred_y = pred_y.reshape(75, )
            print(pred_y)
            cla_ind = np.argwhere(pred_y == 1)[:, 0]
            # cla_ind = int(cla_ind)
            print(cla_ind)
            cau_num = len(cla_ind)
            print(cau_num)
            if cau_num == 0:
                self.cause_edt.setPlainText("该句子中不存在原因子句")
            else:
                # 通过训练结果找出原因子句
                inputFile = codecs.open('./data/clause_keywords.csv', 'r', 'utf-8')
                i = 0
                sen_id_all, cla_id_all, clause_all = [], [], []
                for line in inputFile.readlines():
                    line = line.strip().split(',')
                    sen_id, cla_id, clause = int(line[0]), int(line[1]), line[-1]
                    if sen_id == senID:
                        for i in range(0, cau_num):
                            if cla_id == cla_ind[i]:
                                clause_all.append(clause.replace(' ', ''))
                                self.cause_edt.setPlainText(clause.replace(' ', ''))
                # clause_all = np.array(clause_all)
                # print(sen_id_all.shape)
                # print(cla_id_all.shape)
                print(clause_all)
                inputFile.close()

    # def Ecc(self):
    #     clause = self.clause_edt.toPlainText()
    #     if clause == "":
    #         QMessageBox.warning(self, "警告", "请输入句子")
    #     else:
    #         # 生成测试集
    #
    #         # reload训练好的模型
    #
    #         # 将测试集传入模型进行训练
    #
    #         # 通过训练结果找出原因子句
    #
    #         #打印原因子句
    #         self.cause_edt.setPlainText(clause)




if __name__ == '__main__':
    app = QApplication(sys.argv)
    MainWindow = MainWindow()
    MainWindow.show()
    sys.exit(app.exec_())

