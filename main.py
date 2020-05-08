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
import re
import jieba
from sklearn.model_selection import KFold
from sklearn.model_selection import ParameterGrid

max_doc_len = 75
max_sen_len = 45
embedding_dim = 200

FLAGS = tf.app.flags.FLAGS

class MainWindow(QtWidgets.QMainWindow,Ui_MainWindow):
    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent=parent)
        self.setupUi(self)

    def Clear(self):
        self.clause_edt.clear()
        self.emotion_edt.clear()
        self.cause_edt.clear()

    def New(self):
        senID = random.randint(1, 2105)
        the_line = linecache.getline('./data/datacsv_2105.csv', senID)
        the_line = the_line.strip().split(',')
        the_line = the_line[2]

        self.clause_edt.setPlainText(the_line)

        # 生成测试集
        #func.load_data()：return x, y_position, y, sen_len, doc_len, relative_pos, relative_pos_a, embedding, embedding_pos
        x_data, y_position_data, y_data, sen_len_data, doc_len_data, word_distance, word_distance_a, word_distance_e, word_embedding, pos_embedding, pos_embedding_a, pos_embedding_e = func.load_data()

        print("senID:{}\n".format(senID))

        te_x = x_data[senID]
        te_x = te_x[np.newaxis, :]
        te_sen_len = sen_len_data[senID]
        te_sen_len = te_sen_len[np.newaxis, :]
        te_doc_len = np.array([doc_len_data[senID]])
        te_word_dis = word_distance_e[senID]
        te_word_dis = te_word_dis[np.newaxis, :]

        print("te_x.shape:{}\n".format(te_x.shape))
        print("te_sen_len.shape:{}\n".format(te_sen_len.shape))
        print("te_doc_len:{}\n".format(te_doc_len))
        print("te_doc_len.shape:{}\n".format(te_doc_len.shape))
        print("te_word_dis.shape:{}\n".format(te_word_dis.shape))

        test = [te_x, te_sen_len, te_doc_len, te_word_dis, 1., 1., word_embedding]
        tf.reset_default_graph()

        with tf.Session() as sess:
            # reload训练好的模型

            saver = tf.train.import_meta_graph('run_final_ee/model.ckpt-13.meta')
            model_file = tf.train.latest_checkpoint('run_final_ee/')
            saver.restore(sess, model_file)

            tenboard_dir = './tensorboard/RTHN_EE'
            graph = tf.get_default_graph()
            writer = tf.summary.FileWriter(tenboard_dir, graph)
            writer.add_graph(sess.graph)

            x = graph.get_tensor_by_name("x:0")
            word_dis = graph.get_tensor_by_name("word_dis:0")
            sen_len = graph.get_tensor_by_name("sen_len:0")
            doc_len = graph.get_tensor_by_name("doc_len:0")
            keep_prob1 = graph.get_tensor_by_name("keep_prob1:0")
            keep_prob2 = graph.get_tensor_by_name("keep_prob2:0")
            pred_y_op = graph.get_tensor_by_name("pred_y_op:0")
            pred_pos_op = graph.get_tensor_by_name("pred_pos_op:0")
            word_embedding = graph.get_tensor_by_name("word_embedding:0")

            placeholders = [x, sen_len, doc_len, word_dis, keep_prob1, keep_prob2, word_embedding]
            # placeholders = [x, y, sen_len, doc_len, word_dis, keep_prob1, keep_prob2]

            # 将测试集传入模型进行训练
            pred_pos, pred_y = sess.run([pred_pos_op, pred_y_op], feed_dict=dict(zip(placeholders, test)))
            # print("pred_pos.shape:{}",pred_pos.shape)
            # print("pred_y.shape:{}",pred_y.shape)
            pred_pos = pred_pos.reshape(75, )
            pred_y = pred_y.reshape(75, )
            print("pred_pos:{}",pred_pos)
            print("pred_y:{}",pred_y)
            emo_ind = np.argwhere(pred_pos == 1)[:, 0]
            cla_ind = np.argwhere(pred_y == 1)[:, 0]
            # cla_ind = int(cla_ind)
            print("emo_ind:{}".format(emo_ind))
            print("cla_ind:{}".format(cla_ind))
            emo_num = len(emo_ind)
            cau_num = len(cla_ind)
            print("emo_num:{}".format(emo_num))
            print("cau_num:{}".format(cau_num))

            if cau_num == 0:
                self.cause_edt.setPlainText("该句子中不存在原因子句")
            else:
                # 通过训练结果找出原因子句
                self.cause_edt.clear()
                inputFile1 = codecs.open('./data/clause_keywords.csv', 'r', 'utf-8')
                i = 0
                clause_all = []
                for line in inputFile1.readlines():
                    line = line.strip().split(',')
                    sen_id, cla_id, clause = int(line[0]), int(line[1]), line[-1]
                    if sen_id == senID:
                        clause_all.append(clause.replace(' ', ''))
                clause_all = np.array(clause_all)
                for i in range(0, cau_num):
                    self.cause_edt.append(clause_all[cla_ind[i]])
    #                 self.cause_edt.setPlainText(clause.replace(' ', ''))
                # print(clause_all)
                inputFile1.close()

            if emo_num == 0:
                self.emotion_edt.setPlainText("该句子中不存在情感子句")
            else:
                # 通过训练结果找出情感子句
                self.emotion_edt.clear()
                inputFile2 = codecs.open('./data/clause_keywords.csv', 'r', 'utf-8')
                i = 0
                clause_e_all = []
                for line in inputFile2.readlines():
                    line = line.strip().split(',')
                    sen_id, cla_id, clause = int(line[0]), int(line[1]), line[-1]
                    if sen_id == senID:
                        for j in range(0, emo_num):
                            if cla_id == emo_ind[i] + 1:
                                self.emotion_edt.append(clause.replace(' ', ''))
                #                 self.emotion_edt.setPlainText(clause.replace(' ', ''))
                # print(clause_e_all)
                inputFile2.close()

    def Ecc(self):
        clause = self.clause_edt.toPlainText()
        if clause == "":
            QMessageBox.warning(self, "警告", "请输入句子")
        else:
            # 生成测试集
            the_line = self.clause_edt.toPlainText()
            print(the_line)
            # the_line = "“当我看到建议被采纳，部委领导写给我的回信时，我知道我正在为这个国家的发展尽着一份力量。”27日，河北省邢台钢铁有限公司的普通工人白金跃，拿着历年来国家各部委反馈给他的感谢信，激动地对中新网记者说。“27年来，国家公安部、国家工商总局、国家科学技术委员会科技部、卫生部、国家发展改革委员会等部委均接受并采纳过的我的建议。”"

            # 找到所有不重复词的集合
            words, clauses = [], []
            pattern = r',|\.|/|;|\'|`|\[|\]|<|>|\?|:|"|\{|\}|\~|!|@|#|\$|%|\^|&|\(|\)|-|=|\_|\+|，|。|；|‘|’|【|】|·|！| |…|（|）|“|”'
            sentence = re.split(pattern, the_line)
            while '' in sentence:
                sentence.remove('')
            print("sentence:{}".format(sentence))
            for items in sentence:
                c_words = jieba.cut(items)
                c_words = list(c_words)
                while '、' in c_words:
                    c_words.remove('、')
                clauses.append(c_words)
                words.extend(c_words)

            print("clauses:{}".format(clauses))
            words = set(words)  # 所有不重复词的集合
            print(words)
            word_idx = dict((c, k + 1) for k, c in enumerate(words))  # 给所有词编号(字典)
            # 生成词嵌入字典
            w2v = {}
            inputFile = codecs.open('./data/w2v_200.txt', 'r', 'utf-8')
            for line in inputFile.readlines():
                line = line.strip().split(' ')
                w, ebd = line[0], line[1:]
                w2v[w] = ebd  # 每个字符对应词向量
            inputFile.close()
            embedding = [list(np.zeros(embedding_dim))]
            # 生成词嵌入
            hit = 0
            for item in words:
                if item in w2v:
                    vec = list(map(float, w2v[item]))
                    hit += 1
                else:
                    vec = list(np.random.rand(embedding_dim) / 5. - 0.1)  # 从均匀分布[-0.1,0.1]中随机取
                embedding.append(vec)
            embedding = np.array(embedding)
            print("embedding.shape:{}".format(embedding.shape))

            te_x, te_sen_len, te_doc_len, te_word_dis = [], [], [], []
            clause_all, tmp_clause_len, relative_pos_e, relative_pos_all_e = [], [], [], []

            for index, words in enumerate(clauses):
                clause = [0] * max_sen_len
                relative_pos_clause_e = [0] * max_sen_len
                for i, word in enumerate(words):
                    clause[i] = int(word_idx[word])
                    relative_pos_clause_e[i] = 1
                relative_pos_all_e.append(np.array(relative_pos_clause_e))
                clause_all.append(np.array(clause))
                tmp_clause_len.append(len(words))
            te_doc_len.append(len(clause_all))
            for j in range(max_doc_len - len(clause_all)):
                clause_all.append(np.zeros((max_sen_len,)))
                tmp_clause_len.append(0)
                relative_pos_all_e.append(np.zeros((max_sen_len,)))
            relative_pos_e.append(relative_pos_all_e)
            te_x.append(clause_all)
            te_sen_len.append(tmp_clause_len)

            te_word_dis, te_x, te_sen_len, te_doc_len = map(np.array, [relative_pos_e, te_x, te_sen_len, te_doc_len])
            # print("te_word_dis.shape:{}".format(te_word_dis.shape))
            # print("te_word_dis[0][0]:{}".format(te_word_dis[0][0]))
            # print("te_x.shape:{}".format(te_x.shape))
            # print("te_x[0][0]:{}".format(te_x[0][0]))
            # print("te_doc_len.shape:{}".format(te_doc_len.shape))
            # print("te_doc_len:{}".format(te_doc_len))
            # print("te_sen_len.shape:{}".format(te_sen_len.shape))
            # print("te_sen_len:{}".format(te_sen_len))

            te_x, te_sen_len, te_doc_len, te_word_dis = map(np.array, [te_x, te_sen_len, te_doc_len, te_word_dis])
            test = [te_x, te_sen_len, te_doc_len, te_word_dis, 1., 1., embedding]

            # 传入测试集进行训练
            tf.reset_default_graph()
            with tf.Session() as sess:
                # reload训练好的模型
                saver = tf.train.import_meta_graph('run_final_ee/model.ckpt-13.meta')
                model_file = tf.train.latest_checkpoint('run_final_ee/')
                saver.restore(sess, model_file)

                graph = tf.get_default_graph()

                x = graph.get_tensor_by_name("x:0")
                word_dis = graph.get_tensor_by_name("word_dis:0")
                sen_len = graph.get_tensor_by_name("sen_len:0")
                doc_len = graph.get_tensor_by_name("doc_len:0")
                keep_prob1 = graph.get_tensor_by_name("keep_prob1:0")
                keep_prob2 = graph.get_tensor_by_name("keep_prob2:0")
                pred_y_op = graph.get_tensor_by_name("pred_y_op:0")
                pred_pos_op = graph.get_tensor_by_name("pred_pos_op:0")
                word_embedding = graph.get_tensor_by_name("word_embedding:0")

                placeholders = [x, sen_len, doc_len, word_dis, keep_prob1, keep_prob2, word_embedding]

                # 将测试集传入模型进行训练
                pred_pos, pred_y = sess.run([pred_pos_op, pred_y_op], feed_dict=dict(zip(placeholders, test)))
                # print("pred_pos.shape:{}".format(pred_pos.shape))
                # print("pred_y.shape:{}".format(pred_y.shape))
                pred_pos = pred_pos.reshape(75, )
                pred_y = pred_y.reshape(75, )
                print("pred_pos:{}".format(pred_pos))
                print("pred_y:{}".format(pred_y))
                emo_ind = np.argwhere(pred_pos == 1)[:, 0]
                cla_ind = np.argwhere(pred_y == 1)[:, 0]
                # cla_ind = int(cla_ind)
                print("emo_ind:{}".format(emo_ind))
                print("cla_ind:{}".format(cla_ind))
                emo_num = len(emo_ind)
                cau_num = len(cla_ind)
                print("emo_num:{}".format(emo_num))
                print("cau_num:{}".format(cau_num))

                result_c, result_e = [], []
                if cau_num == 0:
                    self.cause_edt.setPlainText("该句子中不存在原因子句")
                else:
                    # 通过训练结果找出原因子句
                    self.cause_edt.clear()
                    for i in range(0, cau_num):
                        self.cause_edt.append(sentence[cla_ind[i]])
                    #     # self.cause_edt.setPlainText(clause.replace(' ', ''))
                    # print("result_c:{}".format(result_c))
                if emo_num == 0:
                    self.emotion_edt.setPlainText("该句子中不存在情感子句")
                    # print("该句子中不存在情感子句")
                else:
                    self.emotion_edt.clear()
                    # 通过训练结果找出情感子句
                    for i in range(0, emo_num):
                        self.emotion_edt.append(sentence[emo_ind[i]])
                    #     # self.cause_edt.setPlainText(clause.replace(' ', ''))
                    # print("result_e:{}".format(result_e))

if __name__ == '__main__':
    app = QApplication(sys.argv)
    MainWindow = MainWindow()
    MainWindow.show()
    sys.exit(app.exec_())

    # the_line = "“当我看到建议被采纳，部委领导写给我的回信时，我知道我正在为这个国家的发展尽着一份力量。”27日，河北省邢台钢铁有限公司的普通工人白金跃，拿着历年来国家各部委反馈给他的感谢信，激动地对中新网记者说。“27年来，国家公安部、国家工商总局、国家科学技术委员会科技部、卫生部、国家发展改革委员会等部委均接受并采纳过的我的建议。”"
    #
    # # 找到所有不重复词的集合
    # words, clauses = [], []
    # pattern = r',|\.|/|;|\'|`|\[|\]|<|>|\?|:|"|\{|\}|\~|!|@|#|\$|%|\^|&|\(|\)|-|=|\_|\+|，|。|；|‘|’|【|】|·|！| |…|（|）|“|”'
    # sentence = re.split(pattern, the_line)
    # while '' in sentence:
    #     sentence.remove('')
    # print("sentence:{}".format(sentence))
    # for items in sentence:
    #     c_words = jieba.cut(items)
    #     c_words = list(c_words)
    #     while '、' in c_words:
    #         c_words.remove('、')
    #     clauses.append(c_words)
    #     words.extend(c_words)
    #
    # print("clauses:{}".format(clauses))
    # words = set(words)  # 所有不重复词的集合
    # print(words)
    # word_idx = dict((c, k + 1) for k, c in enumerate(words))  # 给所有词编号(字典)
    # # 生成词嵌入字典
    # w2v = {}
    # inputFile = codecs.open('./data/w2v_200.txt', 'r', 'utf-8')
    # for line in inputFile.readlines():
    #     line = line.strip().split(' ')
    #     w, ebd = line[0], line[1:]
    #     w2v[w] = ebd  # 每个字符对应词向量
    # inputFile.close()
    # embedding = [list(np.zeros(embedding_dim))]
    # # 生成词嵌入
    # hit = 0
    # for item in words:
    #     if item in w2v:
    #         vec = list(map(float, w2v[item]))
    #         hit += 1
    #     else:
    #         vec = list(np.random.rand(embedding_dim) / 5. - 0.1)  # 从均匀分布[-0.1,0.1]中随机取
    #     embedding.append(vec)
    # embedding = np.array(embedding)
    # print("embedding.shape:{}".format(embedding.shape))
    #
    # te_x, te_sen_len, te_doc_len, te_word_dis = [], [], [], []
    # clause_all, tmp_clause_len, relative_pos_e, relative_pos_all_e = [], [], [], []
    #
    # for index, words in enumerate(clauses):
    #     clause = [0] * max_sen_len
    #     relative_pos_clause_e = [0] * max_sen_len
    #     for i, word in enumerate(words):
    #         clause[i] = int(word_idx[word])
    #         relative_pos_clause_e[i] = 1
    #     relative_pos_all_e.append(np.array(relative_pos_clause_e))
    #     clause_all.append(np.array(clause))
    #     tmp_clause_len.append(len(words))
    # te_doc_len.append(len(clause_all))
    # for j in range(max_doc_len - len(clause_all)):
    #     clause_all.append(np.zeros((max_sen_len,)))
    #     tmp_clause_len.append(0)
    #     relative_pos_all_e.append(np.zeros((max_sen_len,)))
    # relative_pos_e.append(relative_pos_all_e)
    # te_x.append(clause_all)
    # te_sen_len.append(tmp_clause_len)
    #
    # te_word_dis, te_x, te_sen_len, te_doc_len = map(np.array, [relative_pos_e, te_x, te_sen_len, te_doc_len])
    # # print("te_word_dis.shape:{}".format(te_word_dis.shape))
    # # print("te_word_dis[0][0]:{}".format(te_word_dis[0][0]))
    # # print("te_x.shape:{}".format(te_x.shape))
    # # print("te_x[0][0]:{}".format(te_x[0][0]))
    # # print("te_doc_len.shape:{}".format(te_doc_len.shape))
    # # print("te_doc_len:{}".format(te_doc_len))
    # # print("te_sen_len.shape:{}".format(te_sen_len.shape))
    # # print("te_sen_len:{}".format(te_sen_len))
    #
    # te_x, te_sen_len, te_doc_len, te_word_dis = map(np.array, [te_x, te_sen_len, te_doc_len, te_word_dis])
    # test = [te_x, te_sen_len, te_doc_len, te_word_dis, 1., 1., embedding]
    #
    # # 传入测试集进行训练
    # tf.reset_default_graph()
    # with tf.Session() as sess:
    #     # reload训练好的模型
    #     saver = tf.train.import_meta_graph('run_final_ee/model.ckpt-14.meta')
    #     model_file = tf.train.latest_checkpoint('run_final_ee/')
    #     saver.restore(sess, model_file)
    #
    #     graph = tf.get_default_graph()
    #
    #     x = graph.get_tensor_by_name("x:0")
    #     word_dis = graph.get_tensor_by_name("word_dis:0")
    #     sen_len = graph.get_tensor_by_name("sen_len:0")
    #     doc_len = graph.get_tensor_by_name("doc_len:0")
    #     keep_prob1 = graph.get_tensor_by_name("keep_prob1:0")
    #     keep_prob2 = graph.get_tensor_by_name("keep_prob2:0")
    #     pred_y_op = graph.get_tensor_by_name("pred_y_op:0")
    #     pred_pos_op = graph.get_tensor_by_name("pred_pos_op:0")
    #     word_embedding = graph.get_tensor_by_name("word_embedding:0")
    #
    #     placeholders = [x, sen_len, doc_len, word_dis, keep_prob1, keep_prob2, word_embedding]
    #
    #     # 将测试集传入模型进行训练
    #     pred_pos, pred_y = sess.run([pred_pos_op, pred_y_op], feed_dict=dict(zip(placeholders, test)))
    #     print("pred_pos.shape:{}".format(pred_pos.shape))
    #     print("pred_y.shape:{}".format(pred_y.shape))
    #     pred_pos = pred_pos.reshape(75, )
    #     pred_y = pred_y.reshape(75, )
    #     print("pred_pos".format(pred_pos))
    #     print("pred_y".format(pred_y))
    #     emo_ind = np.argwhere(pred_pos == 1)[:, 0]
    #     cla_ind = np.argwhere(pred_y == 1)[:, 0]
    #     # cla_ind = int(cla_ind)
    #     print("emo_ind:{}".format(emo_ind))
    #     print("cla_ind:{}".format(cla_ind))
    #     emo_num = len(emo_ind)
    #     cau_num = len(cla_ind)
    #     print("emo_num:{}".format(emo_num))
    #     print("cau_num:{}".format(cau_num))
    #
    #     result_c, result_e = [], []
    #     if cau_num == 0:
    #         # self.cause_edt.setPlainText("该句子中不存在原因子句")
    #         print("该句子中不存在原因子句")
    #     else:
    #         # 通过训练结果找出原因子句
    #         # self.cause_edt.clear()
    #         for i in range(0, cau_num):
    #             # self.cause_edt.append(sentence[cla_ind[i]])
    #         #     # self.cause_edt.setPlainText(clause.replace(' ', ''))
    #             print("sentence[cla_ind[i]]:{}".format(sentence[cla_ind[i]]))
    #     if emo_num == 0:
    #         # self.emotion_edt.setPlainText("该句子中不存在情感子句")
    #         print("该句子中不存在情感子句")
    #     else:
    #         # self.emotion_edt.clear()
    #         # 通过训练结果找出情感子句
    #         for i in range(0, emo_num):
    #             # self.emotion_edt.append(sentence[emo_ind[i]])
    #         #     # self.cause_edt.setPlainText(clause.replace(' ', ''))
    #             print("sentence[emo_ind[i]]:{}".format(sentence[emo_ind[i]]))