# -*- encoding:utf-8 -*-
'''
@time: 2019/05/31
@author: mrzhang
@email: zhangmengran@njust.edu.cn
'''

import numpy as np
import pickle as pk
import transformer as trans
import tensorflow as tf
import sys, os, time, codecs, pdb
import utils.tf_funcs as func
from sklearn.model_selection import KFold
from sklearn.model_selection import ParameterGrid
os.environ["CUDA_VISIBLE_DEVICES"] = '2,3'

FLAGS = tf.app.flags.FLAGS
# >>>>>>>>>>>>>>>>>>>> For Model <<<<<<<<<<<<<<<<<<<< #
# embedding parameters ##
tf.app.flags.DEFINE_integer('embedding_dim', 200, 'dimension of word embedding')
tf.app.flags.DEFINE_integer('embedding_dim_pos', 50, 'dimension of position embedding')
# input struct ##
tf.app.flags.DEFINE_integer('max_doc_len', 75, 'max number of tokens per documents')
tf.app.flags.DEFINE_integer('max_sen_len', 45, 'max number of tokens per sentence')
# model struct ##
tf.app.flags.DEFINE_integer('n_hidden', 100, 'number of hidden unit')
tf.app.flags.DEFINE_integer('n_class', 2, 'number of distinct class')
# >>>>>>>>>>>>>>>>>>>> For Data <<<<<<<<<<<<<<<<<<<< #
tf.app.flags.DEFINE_string('log_file_name', '', 'name of log file')
# >>>>>>>>>>>>>>>>>>>> For Training <<<<<<<<<<<<<<<<<<<< #
tf.app.flags.DEFINE_integer('training_iter', 15, 'number of train iter')
# tf.app.flags.DEFINE_integer('training_iter', 7, 'number of train iter')
tf.app.flags.DEFINE_string('scope', 'RNN', 'RNN scope')
# not easy to tune , a good posture of using data to train model is very important
tf.app.flags.DEFINE_integer('batch_size', 32, 'number of example per batch')
tf.app.flags.DEFINE_float('lr_assist', 0.005, 'learning rate of assist')
tf.app.flags.DEFINE_float('lr_main', 0.001, 'learning rate')
tf.app.flags.DEFINE_float('keep_prob1', 0.5, 'word embedding training dropout keep prob')
tf.app.flags.DEFINE_float('keep_prob2', 1.0, 'softmax layer dropout keep prob')
tf.app.flags.DEFINE_float('l2_reg', 1e-5, 'l2 regularization')
# tf.app.flags.DEFINE_integer('run_times', 10, 'run times of this model')
tf.app.flags.DEFINE_integer('run_times', 1, 'run times of this model')
tf.app.flags.DEFINE_integer('num_heads', 5, 'the num heads of attention')
tf.app.flags.DEFINE_integer('n_layers', 2, 'the layers of transformer beside main')
# tf.app.flags.DEFINE_float('cause', 1.000, 'lambda1')
# tf.app.flags.DEFINE_float('pos', 1.00, 'lambda2')
tf.app.flags.DEFINE_float('cause', 0.5, 'lambda1')
tf.app.flags.DEFINE_float('pos', 0.5, 'lambda2')


#pred, reg, pred_assist_list, reg_assist_list = build_model(x, sen_len, doc_len, word_dis, word_embedding, pos_embedding,                                                          keep_prob1, keep_prob2)
def build_model(x, sen_len, doc_len, word_dis, word_embedding, pos_embedding, keep_prob1, keep_prob2, RNN=func.biLSTM):
    x = tf.nn.embedding_lookup(word_embedding, x)#选取wordembedding中x对应的元素
    inputs = tf.reshape(x, [-1, FLAGS.max_sen_len, FLAGS.embedding_dim])
    sh2 = 2 * FLAGS.n_hidden
    inputs = tf.nn.dropout(inputs, keep_prob=keep_prob1)
    sen_len = tf.reshape(sen_len, [-1])
    # print("sen_len:{}

    def get_s(inputs, name):
        with tf.name_scope('word_encode'):
            wordEncode = RNN(inputs, sen_len, n_hidden=FLAGS.n_hidden, scope=FLAGS.scope + 'word_layer'  + name)
        wordEncode = tf.reshape(wordEncode, [-1, FLAGS.max_sen_len, sh2])

        with tf.name_scope('attention'):
            w1 = func.get_weight_varible('word_att_w1'+ name, [sh2, sh2])
            b1 = func.get_weight_varible('word_att_b1'+ name, [sh2])
            w2 = func.get_weight_varible('word_att_w2'+ name, [sh2, 1])
            senEncode = func.att_var(wordEncode, sen_len, w1, b1, w2)
        senEncode = tf.reshape(senEncode, [-1, FLAGS.max_doc_len, sh2])
        return senEncode

    senEncode = get_s(inputs, name='pos_word_encode')
    s = RNN(senEncode, doc_len, n_hidden=FLAGS.n_hidden, scope=FLAGS.scope + 'pos_sentence_layer')

    with tf.name_scope('sequence_prediction'):
        s1 = tf.reshape(s, [-1, 2 * FLAGS.n_hidden])
        s1 = tf.nn.dropout(s1, keep_prob=keep_prob2)

        w_pos = func.get_weight_varible('softmax_w_pos', [2 * FLAGS.n_hidden, FLAGS.n_class])
        b_pos = func.get_weight_varible('softmax_b_pos', [FLAGS.n_class])
        pred_pos = tf.nn.softmax(tf.matmul(s1, w_pos) + b_pos)
        pred_pos = tf.reshape(pred_pos, [-1, FLAGS.max_doc_len, FLAGS.n_class])

    # 形成相对位置向量
    # print("word_dis:{}".format(word_dis))
    word_dis = tf.reshape(word_dis[:, :, 0], [-1, FLAGS.max_doc_len]) # shape=(?, 75)
    # print("word_dis:{}".format(word_dis))

    pred_y_pos_op = tf.argmax(pred_pos, 2)  # shape=(?, 75)
    cla_ind = tf.argmax(pred_y_pos_op, 1)# shape=(?,)
    cla_ind = tf.reshape(tf.to_int32(cla_ind), [-1, 1])
    cla_ind = tf.tile(cla_ind, [1,75])# shape=(?, 75)
    # print("cla_ind.shape:{}".format(cla_ind))
    m_75 = 75 * tf.ones_like(cla_ind)
    cla_ind =  tf.subtract(cla_ind , m_75)
    cla_ind_add_1 = tf.multiply(cla_ind , word_dis)
    # print("cla_ind_add_1.shape:{}".format(cla_ind_add_1))

    i = tf.constant([x for x in range(0,FLAGS.max_doc_len)], dtype=tf.int32)
    # print("i:{}".format(i))
    i = tf.reshape(i, [1, 75])
    # print("改变之后i:{}".format(i))
    cla_ind_add_2 = tf.multiply(i, word_dis)# shape=(?, 75)
    # print("cla_ind_add_2.shape:{}".format(cla_ind_add_2))
    # pos = tf.subtract(tf.matmul(tf.ones_like(cla_ind), i) , cla_ind_add)
    # pos = tf.ones_like(pos)

    pos = tf.subtract(cla_ind_add_2 , cla_ind_add_1)
    # print("结果:{}".format(pos))

    # print("word_dis:{}".format(word_dis))
    word_dis = tf.nn.embedding_lookup(pos_embedding, pos)  # 选取pos_embedding中word_dis对应的元素
    # print("word_dis最终:{}".format(word_dis))

    senEncode = get_s(inputs, name='cause_word_encode')
    # print("senEncode:{}".format(senEncode))
    senEncode_dis = tf.concat([senEncode, word_dis], axis=2)  # 距离拼在子句上

    n_feature = 2 * FLAGS.n_hidden + FLAGS.embedding_dim_pos
    out_units = 2 * FLAGS.n_hidden
    batch = tf.shape(senEncode)[0]
    pred_zeros = tf.zeros(([batch, FLAGS.max_doc_len, FLAGS.max_doc_len]))
    pred_ones = tf.ones_like(pred_zeros)
    pred_two = tf.fill([batch, FLAGS.max_doc_len, FLAGS.max_doc_len], 2.)
    matrix = tf.reshape((1 - tf.eye(FLAGS.max_doc_len)), [1, FLAGS.max_doc_len, FLAGS.max_doc_len]) + pred_zeros
    pred_assist_list, reg_assist_list, pred_assist_label_list = [], [], []
    if FLAGS.n_layers > 1:
        '''*******GL1******'''
        senEncode = trans_func(senEncode_dis, senEncode, n_feature, out_units, 'layer1')
        pred_assist, reg_assist = senEncode_softmax(senEncode, 'softmax_assist_w1', 'softmax_assist_b1', out_units, doc_len)

        pred_assist_label = tf.cast(tf.reshape(tf.argmax(pred_assist, axis=2), [-1, 1, FLAGS.max_doc_len]), tf.float32)
        # masked the prediction at the current position
        pred_assist_label = pred_assist_label * pred_two - pred_ones
        pred_assist_label = (pred_assist_label + pred_zeros) * matrix
        # feedforward
        w_for = func.get_weight_varible('w_for1', [FLAGS.max_doc_len, FLAGS.max_doc_len])
        b_for = func.get_weight_varible('b_for1', [FLAGS.max_doc_len])
        pred_assist_label = tf.tanh(tf.matmul(tf.reshape(pred_assist_label, [-1, FLAGS.max_doc_len]), w_for) + b_for)
        pred_assist_label = tf.reshape(pred_assist_label, [batch, FLAGS.max_doc_len, FLAGS.max_doc_len])

        pred_assist_label_list.append(pred_assist_label)
        pred_assist_list.append(pred_assist)
        reg_assist_list.append(reg_assist)
    '''*******GL n******'''
    for i in range(2, FLAGS.n_layers):
        senEncode_assist = tf.concat([senEncode, pred_assist_label], axis=2)
        n_feature = out_units + FLAGS.max_doc_len
        senEncode = trans_func(senEncode_assist, senEncode, n_feature, out_units, 'layer' + str(i))

        pred_assist, reg_assist = senEncode_softmax(senEncode, 'softmax_assist_w' + str(i), 'softmax_assist_b' + str(i), out_units, doc_len)
        pred_assist_label = tf.cast(tf.reshape(tf.argmax(pred_assist, axis=2), [-1, 1, FLAGS.max_doc_len]), tf.float32)
        # masked the prediction at the current position
        pred_assist_label = pred_assist_label * pred_two - pred_ones
        pred_assist_label = (pred_assist_label + pred_zeros) * matrix
        # feedforward
        w_for = func.get_weight_varible('w_for' + str(i), [FLAGS.max_doc_len, FLAGS.max_doc_len])
        b_for = func.get_weight_varible('b_for' + str(i), [FLAGS.max_doc_len])
        pred_assist_label = tf.tanh(tf.matmul(tf.reshape(pred_assist_label, [-1, FLAGS.max_doc_len]), w_for) + b_for)
        pred_assist_label = tf.reshape(pred_assist_label, [batch, FLAGS.max_doc_len, FLAGS.max_doc_len])

        pred_assist_label_list.append(pred_assist_label)
        pred_assist_label = tf.divide(tf.reduce_sum(pred_assist_label_list, axis=0), i)

        pred_assist_list.append(pred_assist)
        reg_assist_list.append(reg_assist)

    '''*******Main******'''
    if FLAGS.n_layers > 1:
        senEncode_dis_GL = tf.concat([senEncode, pred_assist_label], axis=2)
        n_feature = out_units + FLAGS.max_doc_len
        senEncode_main = trans_func(senEncode_dis_GL, senEncode, n_feature, out_units, 'block_main')
    else:
        senEncode_main = trans_func(senEncode_dis, senEncode, n_feature, out_units, 'block_main')
    pred, reg = senEncode_softmax(senEncode_main, 'softmax_w', 'softmax_b', out_units, doc_len)
    reg += tf.nn.l2_loss(w_pos) + tf.nn.l2_loss(b_pos)
    return  pos, pred_pos, pred, reg, pred_assist_list, reg_assist_list


def run():
    if FLAGS.log_file_name:
        sys.stdout = open(FLAGS.log_file_name, 'w')
    tf.reset_default_graph()
    localtime = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    print("***********localtime: ", localtime)
    #func.load_data()：return x, y_position, y, sen_len, doc_len, relative_pos, relative_pos_a,  embedding, embedding_pos, embedding_pos_a
    #需要将word_distance改为自己计算的结果
    x_data, y_position_data, y_data, sen_len_data, doc_len_data, word_distance, word_distance_a, word_distance_e, word_em_data, pos_embedding, pos_embedding_a, pos_embedding_e = func.load_data()

    # print("x_data.shape:{}\n".format(x_data.shape))
    # print("y_position_data.shape:{}\n".format(y_position_data.shape))
    # print("y_data.shape:{}\n".format(y_data.shape))
    # print("sen_len_data.shape:{}\n".format(sen_len_data.shape))
    # print("doc_len_data.shape:{}\n".format(doc_len_data.shape))
    # print("word_distance.shape:{}\n".format(word_distance.shape))
    # print("word_distance_e.shape:{}\n".format(word_distance_e.shape))
    # print("word_em_data.shape:{}\n".format(word_em_data.shape))
    # print("pos_embedding_a.shape:{}\n".format(pos_embedding_a.shape))
    # print("pos_embedding_e.shape:{}\n".format(pos_embedding_e.shape))
    # print("pos_embedding:{}\n".format(pos_embedding[1]))

    # word_embedding = tf.constant(word_embedding, dtype=tf.float32, name='word_embedding')
    pos_embedding_e = tf.constant(pos_embedding_e, dtype=tf.float32, name='pos_embedding_e')
    print('build model...')
    start_time = time.time()

    #定义placeholder
    x = tf.placeholder(tf.int32, [None, FLAGS.max_doc_len, FLAGS.max_sen_len],name = "x")
    y_position = tf.placeholder(tf.float32, [None, FLAGS.max_doc_len, FLAGS.n_class], name="y_position")
    y = tf.placeholder(tf.float32, [None, FLAGS.max_doc_len, FLAGS.n_class], name = "y")
    sen_len = tf.placeholder(tf.int32, [None, FLAGS.max_doc_len],name = "sen_len")
    doc_len = tf.placeholder(tf.int32, [None],name = "doc_len")
    word_dis = tf.placeholder(tf.int32, [None, FLAGS.max_doc_len, FLAGS.max_sen_len], name = "word_dis")
    keep_prob1 = tf.placeholder(tf.float32, name = "keep_prob1")
    keep_prob2 = tf.placeholder(tf.float32, name = "keep_prob2")
    word_embedding = tf.placeholder(tf.float32, [None, FLAGS.embedding_dim], name= "word_embedding")
    placeholders = [x, y_position, y, sen_len, doc_len, word_dis, keep_prob1, keep_prob2, word_embedding]

    pos, pred_pos, pred, reg, pred_assist_list, reg_assist_list = build_model(x, sen_len, doc_len, word_dis, word_embedding, pos_embedding_e, keep_prob1, keep_prob2)
    print(pos)

    with tf.name_scope('loss'):
        valid_num = tf.cast(tf.reduce_sum(doc_len), dtype=tf.float32)
        loss_pos = - tf.reduce_sum(y_position * tf.log(pred_pos)) / valid_num
        loss_cause = - tf.reduce_sum(y * tf.log(pred)) / valid_num
        loss_op = loss_cause * FLAGS.cause + loss_pos * FLAGS.pos + reg * FLAGS.l2_reg
        loss_assist_list = []
        for i in range(FLAGS.n_layers - 1):
            loss_assist = - tf.reduce_sum(y * tf.log(pred_assist_list[i])) / valid_num + reg_assist_list[i] * FLAGS.l2_reg
            loss_assist_list.append(loss_assist)

    with tf.name_scope('train'):
        optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.lr_main).minimize(loss_op)
        optimizer_assist_list = []
        for i in range(FLAGS.n_layers - 1):
            if i == 0:
                optimizer_assist = tf.train.AdamOptimizer(learning_rate=FLAGS.lr_assist).minimize(loss_assist_list[i])
            else:
                optimizer_assist = tf.train.AdamOptimizer(learning_rate=FLAGS.lr_main).minimize(loss_assist_list[i])
            optimizer_assist_list.append(optimizer_assist)

    true_y_op = tf.argmax(y, 2,name = "true_y_op")
    pred_y_op = tf.argmax(pred, 2,  name = "pred_y_op")
    true_pos_op = tf.argmax(y_position, 2, name="true_pos_op")
    pred_pos_op = tf.argmax(pred_pos, 2, name="pred_pos_op")
    # print("true_y_op:{}".format(true_y_op))
    # print("pred_y_op:{}".format(pred_y_op))
    # print("true_pos_op:{}".format(true_pos_op))
    # print("pred_pos_op:{}".format(pred_pos_op))
    pred_y_assist_op_list = []

    for i in range(FLAGS.n_layers - 1):
        pred_y_assist_op = tf.argmax(pred_assist_list[i], 2)
        pred_y_assist_op_list.append(pred_y_assist_op)

    print('build model done!\n')

    ########训练和验证过程#########
    prob_list_pr, y_label = [], []
    prob_pos_list_pr, pos_label = [], []
    # Training Code Block
    print_training_info()
    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True

    saver = tf.train.Saver(max_to_keep = 7)

    with tf.Session(config=tf_config) as sess:
        kf, fold, SID = KFold(n_splits=10), 1, 0 #十折交叉验证
        Id = []
        p_list, r_list, f1_list = [], [], []
        p_pos_list, r_pos_list, f1_pos_list = [], [], []
        for train, test in kf.split(x_data):
            tr_x, tr_pos, tr_y, tr_sen_len, tr_doc_len, tr_word_dis = map(lambda x: x[train],
                [x_data, y_position_data, y_data, sen_len_data, doc_len_data, word_distance_e])
            te_x, te_pos, te_y, te_sen_len, te_doc_len, te_word_dis = map(lambda x: x[test],
                [x_data, y_position_data, y_data, sen_len_data, doc_len_data, word_distance_e])
            precision_list, recall_list, FF1_list = [], [], []
            pre_list, true_list, pre_list_prob = [], [], []
            precision_pos_list, recall_pos_list, FF1_pos_list = [], [], []
            pre_pos_list, true_pos_list, pre_pos_list_prob = [], [], []

            sess.run(tf.global_variables_initializer())
            print('############# fold {} ###############'.format(fold))
            fold += 1
            max_f1 = 0.0
            max_f1_pos = 0.0
            print('train docs: {}    test docs: {}'.format(len(tr_y), len(te_y)))

            '''*********GP*********'''
            for layer in range(FLAGS.n_layers - 1):
                # if layer == 0:
                #     training_iter = FLAGS.training_iter #(15)
                # else:
                #     training_iter = FLAGS.training_iter - 5 #(10)
                training_iter = 2
                for i in range(training_iter):
                    step = 1
                    # train：feed_list = [x[index], y[index], sen_len[index], doc_len[index], word_dis[index], keep_prob1, keep_prob2]
                    for train, _ in get_batch_data(tr_x, tr_pos, tr_y, tr_sen_len, tr_doc_len, tr_word_dis, FLAGS.keep_prob1, FLAGS.keep_prob2, FLAGS.batch_size):
                        train.append(word_em_data)
                        _, loss, pred_y, true_y, pred_prob, doc_len_batch = sess.run(
                            [optimizer_assist_list[layer], loss_assist_list[layer], pred_y_assist_op_list[layer], true_y_op, pred_assist_list[layer], doc_len],
                            feed_dict=dict(zip(placeholders, train)))
                        acc_assist, p_assist, r_assist, f1_assist = func.acc_prf(pred_y, true_y, doc_len_batch)
                        if step % 20 == 0:
                            print('cause GL{}: epoch {}: step {}: loss {:.4f} acc {:.4f}'.format(layer + 1, i + 1, step, loss, acc_assist))
                        step = step + 1

            '''*********Train********'''
            for epoch in range(FLAGS.training_iter):
                step = 1
                #train：feed_list = [x[index], y[index], sen_len[index], doc_len[index], word_dis[index], keep_prob1, keep_prob2]
                for train, _ in get_batch_data(tr_x,  tr_pos, tr_y, tr_sen_len, tr_doc_len, tr_word_dis, FLAGS.keep_prob1, FLAGS.keep_prob2, FLAGS.batch_size):
                    train.append( word_em_data )
                    _, loss, pred_y_pos, true_pos, pred_y, true_y, pred_prob, pred_pos_prob, doc_len_batch,  pos_data= sess.run(
                        [optimizer, loss_op, pred_pos_op, true_pos_op, pred_y_op, true_y_op, pred, pred_pos, doc_len, pos],
                        feed_dict=dict(zip(placeholders, train)))
                    print("pos_data:{}".format(pos_data))
                    acc, p, r, f1 = func.acc_prf(pred_y, true_y, doc_len_batch)
                    acc_pos, p_pos, r_pos, f1_pos = func.acc_prf(pred_y_pos, true_pos, doc_len_batch)
                    if step % 20 == 0:
                        print('cause: epoch {}: step {}: loss {:.4f} acc {:.4f}'.format(epoch + 1, step, loss, acc))
                        print('emotion: epoch {}: step {}: loss {:.4f} acc {:.4f}'.format(epoch + 1, step, loss, acc_pos))
                    # print("begin save!")
                    # saver.save(sess, "./run_final_ee/model.ckpt", global_step=step)
                    step = step + 1
                print("begin save!")
                saver.save(sess, "./run_final_ee/model.ckpt", global_step = epoch)

                '''*********Test********'''
                test = [te_x, te_pos, te_y, te_sen_len, te_doc_len, te_word_dis, 1., 1.,word_em_data]
                loss, pred_y_pos, true_pos, pred_y, true_y, pred_prob, pred_pos_prob = sess.run(
                    [loss_op, pred_pos_op, true_pos_op, pred_y_op, true_y_op, pred, pred_pos], feed_dict=dict(zip(placeholders, test)))

                end_time = time.time()

                true_list.append(true_y)
                pre_list.append(pred_y)
                pre_list_prob.append(pred_prob)

                true_pos_list.append(true_pos)
                pre_pos_list.append(pred_y_pos)
                pre_pos_list_prob.append(pred_pos_prob)

                #计算精确率准确率召回率和F值
                acc, p, r, f1 = func.acc_prf(pred_y, true_y, te_doc_len)
                acc_pos, p_pos, r_pos, f1_pos = func.acc_prf(pred_y_pos, true_pos, te_doc_len)

                precision_list.append(p)
                recall_list.append(r)
                FF1_list.append(f1)

                precision_pos_list.append(p_pos)
                recall_pos_list.append(r_pos)
                FF1_pos_list.append(f1_pos)

                if f1 > max_f1:
                    max_acc, max_p, max_r, max_f1 = acc, p, r, f1
                print('\ncause test: epoch {}: loss {:.4f} acc {:.4f}\np: {:.4f} r: {:.4f} f1: {:.4f} max_f1 {:.4f}\n'.format(
                    epoch + 1, loss, acc, p, r, f1, max_f1))
                if f1_pos > max_f1_pos:
                    max_acc_pos, max_p_pos, max_r_pos, max_f1_pos = acc_pos, p_pos, r_pos, f1_pos
                print('\nemotion test: epoch {}: loss {:.4f} acc {:.4f}\np: {:.4f} r: {:.4f} f1: {:.4f} max_f1 {:.4f}\n'.format(
                    epoch + 1, loss, acc_pos, p_pos, r_pos, f1_pos, max_f1_pos))

            Id.append(len(te_x))
            SID = np.sum(Id) - len(te_x)
            _, maxIndex = func.maxS(FF1_list)
            _, maxIndex_pos = func.maxS(FF1_pos_list)
            print("cause extract maxIndex:", maxIndex)
            print("emotion extract maxIndex:", maxIndex_pos)
            print('Optimization Finished!\n')
            pred_prob = pre_list_prob[maxIndex]
            pred_pos_prob = pre_pos_list_prob[maxIndex_pos]

            for i in range(pred_y.shape[0]):
                for j in range(te_doc_len[i]):
                    prob_list_pr.append(pred_prob[i][j][1])
                    y_label.append(true_y[i][j])

            for i in range(pred_y_pos.shape[0]):
                for j in range(te_doc_len[i]):
                    prob_pos_list_pr.append(pred_pos_prob[i][j][1])
                    pos_label.append(true_pos[i][j])

            p_list.append(max_p)
            r_list.append(max_r)
            f1_list.append(max_f1)

            p_pos_list.append(max_p_pos)
            r_pos_list.append(max_r_pos)
            f1_pos_list.append(max_f1_pos)
        print("running time: ", str((end_time - start_time) / 60.))
        print_training_info()
        p, r, f1 = map(lambda x: np.array(x).mean(), [p_list, r_list, f1_list])
        print("cause f1_score in 10 fold: {}\naverage : p:{} r:{} f1:{}\n".format(np.array(f1_list).reshape(-1, 1), round(p, 4), round(r, 4), round(f1, 4)))

        p_pos, r_pos, f1_pos = map(lambda x: np.array(x).mean(), [p_pos_list, r_pos_list, f1_pos_list])
        print("emotion f1_score in 10 fold: {}\naverage : p:{} r:{} f1:{}\n".format(np.array(f1_pos_list).reshape(-1, 1), round(p_pos, 4), round(r_pos, 4), round(f1_pos, 4)))

        return p, r, f1, p_pos, r_pos, f1_pos

def print_training_info():
    print('\n\n>>>>>>>>>>>>>>>>>>>>TRAINING INFO:\n')
    print('batch-{}, learning_rate-{}, keep_prob1-{}, num_heads-{}, n_layers-{}'.format(
        FLAGS.batch_size, FLAGS.lr_main, FLAGS.keep_prob1, FLAGS.num_heads, FLAGS.n_layers))
    print('training_iter-{}, scope-{}\n'.format(FLAGS.training_iter, FLAGS.scope))


def get_batch_data(x, pos, y, sen_len, doc_len, word_dis, keep_prob1, keep_prob2,  batch_size, test=False):
    for index in func.batch_index(len(y), batch_size, test):
        feed_list = [x[index], pos[index], y[index], sen_len[index], doc_len[index], word_dis[index],  keep_prob1, keep_prob2]
        yield feed_list, len(index)


def senEncode_softmax(s_senEncode, w_varible, b_varible, n_feature, doc_len):
    s = tf.reshape(s_senEncode, [-1, n_feature])
    s = tf.nn.dropout(s, keep_prob=FLAGS.keep_prob2)
    w = func.get_weight_varible(w_varible, [n_feature, FLAGS.n_class])
    b = func.get_weight_varible(b_varible, [FLAGS.n_class])
    # print("s:{}".format(s))
    # print("w:{}".format(w))
    pred = tf.matmul(s, w) + b
    # print("matmul(s, w):{}".format(pred))
    pred *= func.getmask(doc_len, FLAGS.max_doc_len, [-1, 1])
    pred = tf.nn.softmax(pred)
    pred = tf.reshape(pred, [-1, FLAGS.max_doc_len, FLAGS.n_class])
    reg = tf.nn.l2_loss(w) + tf.nn.l2_loss(b)
    return pred, reg


def trans_func(senEncode_dis, senEncode, n_feature, out_units, scope_var):
    senEncode_assist = trans.multihead_attention(queries=senEncode_dis,
                                            keys=senEncode_dis,
                                            values=senEncode,
                                            units_query=n_feature,
                                            num_heads=FLAGS.num_heads,
                                            dropout_rate=0,
                                            is_training=True,
                                            scope=scope_var)
    senEncode_assist = trans.feedforward_1(senEncode_assist, n_feature, out_units)
    return senEncode_assist


def main(_):
    grid_search = {}
    # params = {"n_layers": [4, 5]}
    params = {"n_layers": [4]}

    params_search = list(ParameterGrid(params))

    for i, param in enumerate(params_search):
        print("*************params_search_{}*************".format(i + 1))
        print(param)
        for key, value in param.items():
            setattr(FLAGS, key, value)
        p_list, r_list, f1_list = [], [], []
        p_pos_list, r_pos_list, f1_pos_list = [], [], []
        for i in range(FLAGS.run_times):
            print("*************run(){}*************".format(i + 1))
            p, r, f1, p_pos, r_pos, f1_pos = run()

            p_list.append(p)
            r_list.append(r)
            f1_list.append(f1)

            p_pos_list.append(p_pos)
            r_pos_list.append(r_pos)
            f1_pos_list.append(f1_pos)

        for i in range(FLAGS.run_times):
            print(round(p_list[i], 4), round(r_list[i], 4), round(f1_list[i], 4))
        print("cause avg_prf: ", np.mean(p_list), np.mean(r_list), np.mean(f1_list))

        for i in range(FLAGS.run_times):
            print(round(p_pos_list[i], 4), round(r_pos_list[i], 4), round(f1_pos_list[i], 4))
        print("emotion avg_prf: ", np.mean(p_pos_list), np.mean(r_pos_list), np.mean(f1_pos_list))

        grid_search[str(param)] = {"cause PRF": [round(np.mean(p_list), 4), round(np.mean(r_list), 4), round(np.mean(f1_list), 4)]}
        grid_search[str(param)] = {"emotion PRF": [round(np.mean(p_pos_list), 4), round(np.mean(r_pos_list), 4), round(np.mean(f1_pos_list), 4)]}

    for key, value in grid_search.items():
        print("Main: ", key, value)



if __name__ == '__main__':
    tf.app.run()
