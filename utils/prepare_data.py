# -*- encoding:utf-8 -*-
'''
@time: 2019/05/31
@author: mrzhang
@email: zhangmengran@njust.edu.cn
'''

import codecs
import random
import numpy as np
import pickle as pk
from sklearn.metrics import precision_score, recall_score, f1_score
import pdb
path = '../data/'
max_doc_len = 75
max_sen_len = 45

#word_dict, _, _ = load_w2v(200, 50, path + 'clause_keywords.csv', path + 'w2v_200.txt')
def load_w2v(embedding_dim, embedding_dim_pos, train_file_path, embedding_path):
    print('\nload embedding...')
    words = []
    inputFile1 = codecs.open(train_file_path, 'r', 'utf-8')
    for line in inputFile1.readlines():
        line = line.strip().split(',')
        emotion, clause = line[2], line[-1]
        words.extend([emotion] + clause.split())
    inputFile1.close()
    words = set(words)  # 所有不重复词的集合
    word_idx = dict((c, k + 1) for k, c in enumerate(words)) #给所有词编号(字典)
    # print("word_idx:{}".format(word_idx))

    w2v = {}
    inputFile2 = codecs.open(embedding_path, 'r', 'utf-8')
    for line in inputFile2.readlines():
        line = line.strip().split(' ')
        w, ebd = line[0], line[1:]
        w2v[w] = ebd #每个字符对应词向量
    inputFile2.close()
    embedding = [list(np.zeros(embedding_dim))]

    hit = 0
    for item in words:
        if item in w2v:
            vec = list(map(float, w2v[item]))
            hit += 1
        else:
            vec = list(np.random.rand(embedding_dim) / 5. - 0.1)  # 从均匀分布[-0.1,0.1]中随机取
        embedding.append(vec)
    # print('w2v_file: {}\nall_words: {} hit_words: {}'.format(
    #     embedding_path, len(words), hit))

    embedding_pos = [list(np.zeros(embedding_dim_pos))]
    embedding_pos_a = [list(np.zeros(embedding_dim_pos))]
    embedding_pos.extend([list(np.random.normal(
        loc=0.0, scale=0.1, size=embedding_dim_pos)) for i in range(-68, 34)])

    embedding_pos_a.extend([list(np.random.normal(
        loc=0.0, scale=0.1, size=embedding_dim_pos)) for i in range(0, max_doc_len - 1)])

    embedding_pos_ap = np.zeros([max_doc_len, embedding_dim])
    for pos in range(max_doc_len):
        for i in range(embedding_dim // 2):
            embedding_pos_ap[pos, 2 * i] = np.sin(pos / np.power(10000, 2 * i / embedding_dim))
            embedding_pos_ap[pos, 2 * i + 1] = np.cos(pos / np.power(10000, 2 * i / embedding_dim))

    # embedding, embedding_pos, embedding_pos_e, embedding_pos_a = np.array(embedding), np.array(embedding_pos), np.array(embedding_pos_e), np.array(embedding_pos_a)
    embedding, embedding_pos, embedding_pos_a, embedding_pos_ap = np.array(embedding), np.array(embedding_pos), np.array(embedding_pos_a), np.array(embedding_pos_ap)

    pk.dump(embedding, open(path + 'embedding.txt', 'wb'))
    pk.dump(embedding_pos, open(path + 'embedding_pos.txt', 'wb'))
    pk.dump(embedding_pos_a, open(path + 'embedding_pos_a.txt', 'wb'))
    pk.dump(embedding_pos_ap, open(path + 'embedding_pos_ap.txt', 'wb'))

    print("embedding.shape: {} embedding_pos.shape: {} embedding_pos_a.shape: {}  embedding_pos_ap.shape: {}".format(
        embedding.shape, embedding_pos.shape, embedding_pos_a.shape, embedding_pos_ap.shape))
    print("load embedding done!\n")
    return word_idx, embedding, embedding_pos


#load_data(path + 'clause_keywords.csv', word_dict)
def load_data(input_file, word_idx, max_doc_len=max_doc_len, max_sen_len=max_sen_len):
    print('load data...')
    relative_pos, relative_pos_a, relative_pos_e, x, y, y_position, sen_len, doc_len = [], [], [], [], [], [], [], []

    y_clause_cause, y_clause_emotion, clause_all, tmp_clause_len, relative_pos_all, relative_pos_all_a, relative_pos_all_e= np.zeros((max_doc_len, 2)), np.zeros((max_doc_len, 2)), [], [], [], [], []
    next_ID = 2
    outputFile3 = codecs.open(input_file, 'r', 'utf-8')
    #n_clause:子句数量
    n_clause, yes_clause, no_clause, n_cut = [0] * 4

    for index, line in enumerate(outputFile3.readlines()):

        n_clause += 1
        line = line.strip().split(',')
        #senID:句子id  clause_idx：子句id  emo_word：情感  sen_pos：到情感子句的距离  cause：是否为原因 words：词
        senID, clause_idx, emo_word, sen_pos, cause, words = int(line[0]), int(line[1]), line[2], int(line[3]), line[4], line[5]
        word_pos = sen_pos + 69 #？？？
        if next_ID == senID:  # 数据文件末尾加了一个冗余的文档，会被丢弃，读到了该句子的最后一个子句
            doc_len.append(len(clause_all))

            # print("len(clause_all):{}\n".format(len(clause_all)))#每个句子的子句个数
            # print("clause_all:{}\n".format(clause_all))#句子，有子句个数行
            # print("y_clause_cause:{}\n".format(y_clause_cause))
            # for senID in range (2,10):
            #     print("sen_len:{}\n".format(sen_len))#每个子句的长度，有子句个数行
            #     print("doc_len:{}\n".format(doc_len))#每个句子的长度，有句子个数列

            for j in range(max_doc_len - len(clause_all)):
                clause_all.append(np.zeros((max_sen_len,)))
                tmp_clause_len.append(0)
                relative_pos_all.append(np.zeros((max_sen_len,)))
                relative_pos_all_a.append(np.zeros((max_sen_len,)))
                relative_pos_all_e.append(np.zeros((max_sen_len,)))
            relative_pos.append(relative_pos_all)
            relative_pos_a.append(relative_pos_all_a)
            relative_pos_e.append(relative_pos_all_e)
            x.append(clause_all)
            y_position.append(y_clause_emotion)
            y.append(y_clause_cause)
            sen_len.append(tmp_clause_len)
            y_clause_cause, y_clause_emotion, clause_all, tmp_clause_len, relative_pos_all, relative_pos_all_a, relative_pos_all_e = np.zeros((max_doc_len, 2)), np.zeros((max_doc_len, 2)), [], [], [], [], []
            next_ID = senID + 1

        clause = [0] * max_sen_len
        relative_pos_clause = [0] * max_sen_len
        relative_pos_clause_a = [0] * max_sen_len
        relative_pos_clause_e = [0] * max_sen_len
        for i, word in enumerate(words.split()):
            clause[i] = int(word_idx[word])
            # print("clause[{}]:{}".format(i,clause[i]))
            relative_pos_clause[i] = word_pos
            relative_pos_clause_a[i] = clause_idx - 1
            relative_pos_clause_e[i] = 1
        # print("clause.shape:{}".format(len(clause)))clause:每个子句中词的编号
        relative_pos_all.append(np.array(relative_pos_clause))
        relative_pos_all_a.append(np.array(relative_pos_clause_a))
        relative_pos_all_e.append(np.array(relative_pos_clause_e))
        clause_all.append(np.array(clause))
        tmp_clause_len.append(len(words.split()))
        if cause == 'no': #不是原因子句
            no_clause += 1
            y_clause_cause[clause_idx - 1] = [1, 0]
        else: #是原因子句
            yes_clause += 1
            y_clause_cause[clause_idx - 1] = [0, 1]
        if sen_pos == 0:#是情感子句
            y_clause_emotion[clause_idx - 1] = [0, 1]
        else:#不是情感子句
            y_clause_emotion[clause_idx - 1] = [1, 0]


    outputFile3.close()
    relative_pos, relative_pos_a, relative_pos_e, x, y_position, y, sen_len, doc_len = map(np.array, [relative_pos, relative_pos_a, relative_pos_e, x, y_position, y, sen_len, doc_len])
    #print(y_position[0])

    pk.dump(relative_pos, open(path + 'relative_pos.txt', 'wb'))#相对位置
    pk.dump(relative_pos_a, open(path + 'relative_pos_a.txt', 'wb'))  # 绝对位置
    pk.dump(relative_pos_e, open(path + 'relative_pos_e.txt', 'wb'))  # 辅助生成相对位置
    pk.dump(x, open(path + 'x.txt', 'wb'))#所有子句
    pk.dump(y_position, open(path + 'y_position.txt', 'wb'))  # 是否为情感子句
    pk.dump(y, open(path + 'y.txt', 'wb'))#是否为原因子句
    pk.dump(sen_len, open(path + 'sen_len.txt', 'wb'))
    pk.dump(doc_len, open(path + 'doc_len.txt', 'wb'))

    print('relative_pos.shape {}\nrelative_pos_a.shape {}\nrelative_pos_e.shape {}\nx.shape {} \ny_position.shape {} \ny.shape {} \nsen_len.shape {} \ndoc_len.shape {}\n'.format(
        relative_pos.shape, relative_pos_a.shape, relative_pos_e.shape, x.shape, y_position.shape, y.shape, sen_len.shape, doc_len.shape
    ))
    # print('y_position {}'.format(y_position[0]))
    # print('relative_pos {}'.format(relative_pos[0][0]))
    # print('relative_pos {}'.format(relative_pos[0][1]))
    # word_dis = np.reshape(relative_pos_e[:, :, 0], [-1, max_doc_len])
    # print('word_dis {}'.format(word_dis[0]))
    # print('word_dis {}'.format(word_dis[1]))
    print('load data done!\n')
    return x, y, sen_len, doc_len

#load_w2v: return word_idx, embedding, embedding_pos
word_dict, _, _ = load_w2v(200, 50, path + 'clause_keywords.csv', path + 'w2v_200.txt')
load_data(path + 'clause_keywords.csv', word_dict)
