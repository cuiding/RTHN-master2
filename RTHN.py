# -*- encoding:utf-8 -*-
'''
@author: mrzhang
@time:2018/12/21 10:36
'''

import numpy as np
import tensorflow as tf
import transformer as trans
from sklearn.model_selection import KFold
import sys, os, time, codecs, pdb
import utils.tf_funcs as func
from sklearn.model_selection import KFold
from sklearn.model_selection import ParameterGrid
os.environ["CUDA_VISIBLE_DEVICES"] = '5,4,0'

FLAGS = tf.app.flags.FLAGS
# >>>>>>>>>>>>>>>>>>>> For Model <<<<<<<<<<<<<<<<<<<< #
# embedding parameters ##
tf.app.flags.DEFINE_string('w2v_file', '../data/w2v_200.txt', 'embedding file')
tf.app.flags.DEFINE_integer('embedding_dim', 200, 'dimension of word embedding')
# input struct ##
tf.app.flags.DEFINE_integer('max_doc_len', 75, 'max number of tokens per documents')
tf.app.flags.DEFINE_integer('max_sen_len', 45, 'max number of tokens per sentence')
# model struct ##
tf.app.flags.DEFINE_integer('n_hidden', 100, 'number of hidden unit')
tf.app.flags.DEFINE_integer('n_class', 2, 'number of distinct class')
# >>>>>>>>>>>>>>>>>>>> For Data <<<<<<<<<<<<<<<<<<<< #
tf.app.flags.DEFINE_string('train_file_path', '../data/clause_keywords.csv', 'training file')
tf.app.flags.DEFINE_string('log_file_name', '', 'name of log file')
# >>>>>>>>>>>>>>>>>>>> For Training <<<<<<<<<<<<<<<<<<<< #
tf.app.flags.DEFINE_integer('training_iter', 15, 'number of train iter')
tf.app.flags.DEFINE_integer('n_layers', 2, 'number of train iter')#这是总层数，子层数是n_layers-1
tf.app.flags.DEFINE_string('scope', 'RNN', 'RNN scope')
tf.app.flags.DEFINE_integer('run_times', 1, 'run times of this model')
tf.app.flags.DEFINE_integer('num_heads', 5, 'the num heads of attention')
# not easy to tune , a good posture of using data to train model is very important
tf.app.flags.DEFINE_integer('batch_size', 32, 'number of example per batch')
tf.app.flags.DEFINE_float('learning_rate', 0.005, 'learning rate')
tf.app.flags.DEFINE_float('keep_prob1', 0.5, 'word embedding training dropout keep prob')
tf.app.flags.DEFINE_float('keep_prob2', 1.0, 'softmax layer dropout keep prob')
tf.app.flags.DEFINE_float('l2_reg', 1e-5, 'l2 regularization')


def build_model(word_embedding, x, sen_len, doc_len, keep_prob1, keep_prob2, RNN = func.biLSTM):
    x = tf.nn.embedding_lookup(word_embedding, x)
    inputs = tf.reshape(x, [-1, FLAGS.max_sen_len, FLAGS.embedding_dim])
    inputs = tf.nn.dropout(inputs, keep_prob=keep_prob1)
    sen_len = tf.reshape(sen_len, [-1])
    with tf.name_scope('word_encode'):
        lstm_wordEncode = RNN(inputs, sen_len, n_hidden=FLAGS.n_hidden, scope=FLAGS.scope + 'word_layer')
    lstm_wordEncode = tf.reshape(lstm_wordEncode, [-1, FLAGS.max_sen_len, 2 * FLAGS.n_hidden])
    with tf.name_scope('word_attention'):
        sh2 = 2 * FLAGS.n_hidden
        w1 = func.get_weight_varible('word_att_w1', [sh2, sh2])
        b1 = func.get_weight_varible('word_att_b1', [sh2])
        w2 = func.get_weight_varible('word_att_w2', [sh2, 1])
        s_wordEncode = func.att_var(lstm_wordEncode, sen_len, w1, b1, w2)
    senEncode = tf.reshape(s_wordEncode, [-1, FLAGS.max_doc_len, 2 * FLAGS.n_hidden])

    n_feature = 2 * FLAGS.n_hidden
    out_units = 2 * FLAGS.n_hidden

    for i in range(1, FLAGS.n_layers):
        senEncode = trans_func(senEncode, senEncode, n_feature, out_units, 'layer' + str(i))

    pred, reg = senEncode_softmax(senEncode, 'softmax_w', 'softmax_b', out_units, doc_len)
    return pred, reg


def run():
    if FLAGS.log_file_name:
        sys.stdout = open(FLAGS.log_file_name, 'w')
    tf.reset_default_graph()
    localtime = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    print("***********localtime: ", localtime)
    x_data, y_position_data, y_data, sen_len_data, doc_len_data, word_distance, word_distance_a, word_distance_e, word_embedding, pos_embedding, pos_embedding_a,  pos_embedding_ap = func.load_data()

    word_embedding = tf.constant(word_embedding, dtype=tf.float32, name='word_embedding')
    print('build model...')

    x = tf.placeholder(tf.int32, [None, FLAGS.max_doc_len, FLAGS.max_sen_len],name = "x")
    sen_len = tf.placeholder(tf.int32, [None, FLAGS.max_doc_len],name = "sen_len")
    doc_len = tf.placeholder(tf.int32, [None],name = "doc_len")
    keep_prob1 = tf.placeholder(tf.float32, name = "keep_prob1")
    keep_prob2 = tf.placeholder(tf.float32, name = "keep_prob2")
    y = tf.placeholder(tf.float32, [None, FLAGS.max_doc_len, FLAGS.n_class], name = "y")
    placeholders = [x, sen_len, doc_len, keep_prob1, keep_prob2, y]

    with tf.name_scope('loss'):
        pred, reg = build_model(word_embedding, x, sen_len, doc_len, keep_prob1, keep_prob2)
        valid_num = tf.cast(tf.reduce_sum(doc_len), dtype=tf.float32)
        loss_op = - tf.reduce_sum(y * tf.log(pred)) / valid_num + reg * FLAGS.l2_reg

    with tf.name_scope('train'):
        optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate).minimize(loss_op)

    true_y_op = tf.argmax(y, 2)
    pred_y_op = tf.argmax(pred, 2, name = "pred_y_op")
    print('build model done!\n')

    # Training Code Block
    print_training_info()

    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    with tf.Session(config=tf_config) as sess:
        kf, fold, SID = KFold(n_splits=10), 1, 0
        Id = []
        p_list, r_list, f1_list, loss_list = [], [], [], []
        start_time = time.time()
        for train, test in kf.split(x_data):
            tr_x, tr_y, tr_sen_len, tr_doc_len = map(lambda x: x[train],
                [x_data, y_data, sen_len_data, doc_len_data])
            te_x, te_y, te_sen_len, te_doc_len= map(lambda x: x[test],
                [x_data, y_data, sen_len_data, doc_len_data])

            precision_list, recall_list, FF1_list, l_list = [], [], [], []
            pre_list, true_list, pre_list_prob = [], [], []
            sess.run(tf.global_variables_initializer())
            print('############# fold {} ###############'.format(fold))
            fold += 1
            max_f1, max_f1_rectify = 0.0, 0.0
            print('train docs: {}    test docs: {}'.format(len(tr_y), len(te_y)))
            for epoch in range(FLAGS.training_iter):
                step = 1
                # ************train************
                for train, _ in get_batch_data(tr_x, tr_sen_len, tr_doc_len, FLAGS.keep_prob1, FLAGS.keep_prob2, tr_y, FLAGS.batch_size):
                    _, loss, pred_y, true_y, pred_prob, doc_len_batch = sess.run(
                        [optimizer, loss_op, pred_y_op, true_y_op, pred, doc_len],
                        feed_dict=dict(zip(placeholders, train)))
                    acc, p, r, f1 = func.acc_prf(pred_y, true_y, doc_len_batch)
                    # if step % 5 == 0:
                    #     print('epoch {}: step {}: loss {:.4f} acc {:.4f}'.format(epoch + 1, step, loss, acc))
                    step = step + 1

                # ************test************
                test = [te_x, te_sen_len, te_doc_len, 1., 1., te_y]
                loss, pred_y, true_y, pred_prob = sess.run(
                    [loss_op, pred_y_op, true_y_op, pred], feed_dict=dict(zip(placeholders, test)))
                true_list.append(true_y)
                pre_list.append(pred_y)
                pre_list_prob.append(pred_prob)

                acc, p, r, f1 = func.acc_prf(pred_y, true_y, te_doc_len)
                precision_list.append(p)
                recall_list.append(r)
                FF1_list.append(f1)
                l_list.append(loss)
                if f1 > max_f1:
                    max_acc, max_p, max_r, max_f1 = acc, p, r, f1
                # print('\nepoch {}: loss {:.4f} acc {:.4f}\n\nnorectify: p {:.4f} r {:.4f} f1 {:.4f} max_f1 {:.4f}'.format(
                #     epoch + 1, loss, acc, p, r, f1, max_f1))


            Id.append(len(te_x))
            SID = np.sum(Id) - len(te_x)
            _, maxIndex = func.maxS(FF1_list)
            print('Optimization Finished!\n')
            p_list.append(max_p)
            r_list.append(max_r)
            f1_list.append(max_f1)
            loss_list.extend(l_list)
            # print("loss_list:{}".format(loss_list))

        # print("loss_list.length:{}".format(len(loss_list)))
        los = np.array(loss_list).reshape(10, FLAGS.training_iter)
        # print("los.shape:{}".format(los.shape))
        lo = np.mean(los, axis=0)
        # print("lo.shape:{}".format(lo.shape))
        end_time = time.time()
        print("running time: ", str((end_time - start_time) / 60.))

        print_training_info()
        p, r, f1 = map(lambda x: np.array(x).mean(), [p_list, r_list, f1_list])
        print("f1_score in 10 fold: {}\nloss of training_iter: {}\naverage : {} {} {}\n".format(np.array(f1_list).reshape(-1, 1), lo,round(p, 4), round(r, 4), round(f1, 4)))

        return p, r, f1


def print_training_info():
    print('\n\n>>>>>>>>>>>>>>>>>>>>TRAINING INFO:\n')
    print('batch-{}, lr-{},  kb1-{}, kb2-{}, l2_reg-{}'.format(
        FLAGS.batch_size, FLAGS.learning_rate, FLAGS.keep_prob1, FLAGS.keep_prob2, FLAGS.l2_reg))
    print('training_iter-{}, RTHN\n'.format(FLAGS.training_iter))


def get_batch_data(x, sen_len, doc_len, keep_prob1, keep_prob2, y, batch_size, test=False):
    for index in func.batch_index(len(y), batch_size, test):
        feed_list = [x[index], sen_len[index], doc_len[index], keep_prob1, keep_prob2, y[index]]
        yield feed_list, len(index)

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

def senEncode_softmax(s_senEncode, w_varible, b_varible, n_feature, doc_len):
    s = tf.reshape(s_senEncode, [-1, n_feature])
    s = tf.nn.dropout(s, keep_prob=FLAGS.keep_prob2)
    w = func.get_weight_varible(w_varible, [n_feature, FLAGS.n_class])
    b = func.get_weight_varible(b_varible, [FLAGS.n_class])
    pred = tf.matmul(s, w) + b
    pred *= func.getmask(doc_len, FLAGS.max_doc_len, [-1, 1])
    pred = tf.nn.softmax(pred)
    pred = tf.reshape(pred, [-1, FLAGS.max_doc_len, FLAGS.n_class], name='pred')
    reg = tf.nn.l2_loss(w) + tf.nn.l2_loss(b)
    return pred, reg


def main(_):
    grid_search = {}
    params = {"n_layers": [5,4,3,2]}

    params_search = list(ParameterGrid(params))
    for i, param in enumerate(params_search):
        print("*************params_search_{}*************".format(i + 1))
        print(param)
        for key, value in param.items():
            setattr(FLAGS, key, value)
        p_list, r_list, f1_list = [], [], []
        for i in range(FLAGS.run_times):
            print("*************run(){}*************".format(i + 1))
            p, r, f1 = run()
            p_list.append(p)
            r_list.append(r)
            f1_list.append(f1)

        for i in range(FLAGS.run_times):
            print(round(p_list[i], 4), round(r_list[i], 4), round(f1_list[i], 4))
        print("avg_prf: ", np.mean(p_list), np.mean(r_list), np.mean(f1_list))

        grid_search[str(param)] = {"PRF": [round(np.mean(p_list), 4), round(np.mean(r_list), 4), round(np.mean(f1_list), 4)]}

    for key, value in grid_search.items():
        print("Main: ", key, value)


if __name__ == '__main__':
    tf.app.run()
