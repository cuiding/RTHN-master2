# -*- encoding:utf-8 -*-

import numpy as np
import pickle as pk
import transformer as trans
import tensorflow as tf
import sys, os, time, codecs, pdb
import utils.tf_funcs as func
from sklearn.model_selection import KFold
from sklearn.model_selection import ParameterGrid
os.environ["CUDA_VISIBLE_DEVICES"] = '0, 5'

FLAGS = tf.app.flags.FLAGS
# >>>>>>>>>>>>>>>>>>>> For Model <<<<<<<<<<<<<<<<<<<< #
# embedding parameters ##
tf.app.flags.DEFINE_integer('embedding_dim', 200, 'dimension of word embedding')
# input struct ##
tf.app.flags.DEFINE_integer('max_doc_len', 75, 'max number of tokens per documents')
tf.app.flags.DEFINE_integer('max_sen_len', 45, 'max number of tokens per sentence')
# model struct ##
tf.app.flags.DEFINE_integer('n_hidden', 100, 'number of hidden unit')
tf.app.flags.DEFINE_integer('n_class', 2, 'number of distinct class')
# >>>>>>>>>>>>>>>>>>>> For Data <<<<<<<<<<<<<<<<<<<< #
tf.app.flags.DEFINE_string('log_file_name', '', 'name of log file')
# >>>>>>>>>>>>>>>>>>>> For Training <<<<<<<<<<<<<<<<<<<< #
tf.app.flags.DEFINE_integer('training_iter', 10, 'number of train iter')
tf.app.flags.DEFINE_string('scope', 'RNN', 'RNN scope')
# not easy to tune , a good posture of using data to train model is very important
tf.app.flags.DEFINE_integer('batch_size', 16, 'number of example per batch')
tf.app.flags.DEFINE_float('lr_main', 0.001, 'learning rate')
tf.app.flags.DEFINE_float('keep_prob1', 0.5, 'word embedding training dropout keep prob')
tf.app.flags.DEFINE_float('keep_prob2', 1.0, 'softmax layer dropout keep prob')
tf.app.flags.DEFINE_float('l2_reg', 1e-5, 'l2 regularization')
# tf.app.flags.DEFINE_integer('run_times', 10, 'run times of this model')
tf.app.flags.DEFINE_integer('run_times', 1, 'run times of this model')

def build_model(x, sen_len, doc_len, word_dis, word_embedding, keep_prob1, keep_prob2, RNN=func.biGRU):
    x = tf.nn.embedding_lookup(word_embedding, x)
    inputs = tf.reshape(x, [-1, FLAGS.max_sen_len, FLAGS.embedding_dim])
    inputs = tf.nn.dropout(inputs, keep_prob=keep_prob1)
    sen_len = tf.reshape(sen_len, [-1])

    def get_s(inputs, name):
        with tf.name_scope('word_encode'):
            inputs = RNN(inputs, sen_len, n_hidden=FLAGS.n_hidden, scope=FLAGS.scope+'word_layer' + name)
        with tf.name_scope('word_attention'):
            sh2 = 2 * FLAGS.n_hidden
            w1 = func.get_weight_varible('word_att_w1' + name, [sh2, sh2])
            b1 = func.get_weight_varible('word_att_b1' + name, [sh2])
            w2 = func.get_weight_varible('word_att_w2' + name, [sh2, 1])
            s = func.att_var(inputs,sen_len,w1,b1,w2)
        s = tf.reshape(s, [-1, FLAGS.max_doc_len, 2 * FLAGS.n_hidden])
        return s

    s = get_s(inputs, name='cause_word_encode')
    s1 = tf.reshape(s, [-1, 2 * FLAGS.n_hidden])
    s1 = tf.nn.dropout(s1, keep_prob=keep_prob2)

    w_cause = func.get_weight_varible('softmax_w_cause', [2 * FLAGS.n_hidden, FLAGS.n_class])
    b_cause = func.get_weight_varible('softmax_b_cause', [FLAGS.n_class])
    pred_cause = tf.nn.softmax(tf.matmul(s1, w_cause) + b_cause)
    pred_cause = tf.reshape(pred_cause, [-1, FLAGS.max_doc_len, FLAGS.n_class])

    reg = tf.nn.l2_loss(w_cause) + tf.nn.l2_loss(b_cause)
    return pred_cause, reg

def run():
    if FLAGS.log_file_name:
        sys.stdout = open(FLAGS.log_file_name, 'w')
    tf.reset_default_graph()
    localtime = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    print("***********localtime: ", localtime)
    #func.load_data()：return x, y, sen_len, doc_len, relative_pos, embedding, embedding_pos
    #需要将word_distance改为自己计算的结果
    x_data, y_position_data, y_data, sen_len_data, doc_len_data, word_distance, word_distance_a, word_distance_e, word_embedding, pos_embedding, pos_embedding_a,  pos_embedding_e = func.load_data()

    word_embedding = tf.constant(word_embedding, dtype=tf.float32, name='word_embedding')
    print('build model...')

    start_time = time.time()

    # 定义placeholder
    x = tf.placeholder(tf.int32, [None, FLAGS.max_doc_len, FLAGS.max_sen_len], name="x")
    y = tf.placeholder(tf.float32, [None, FLAGS.max_doc_len, FLAGS.n_class], name="y")
    sen_len = tf.placeholder(tf.int32, [None, FLAGS.max_doc_len], name="sen_len")
    doc_len = tf.placeholder(tf.int32, [None], name="doc_len")
    word_dis = tf.placeholder(tf.int32, [None, FLAGS.max_doc_len, FLAGS.max_sen_len], name="word_dis")
    keep_prob1 = tf.placeholder(tf.float32, name="keep_prob1")
    keep_prob2 = tf.placeholder(tf.float32, name="keep_prob2")
    placeholders = [x, y, sen_len, doc_len, word_dis, keep_prob1, keep_prob2]

    pred, reg = build_model(x, sen_len, doc_len, word_dis, word_embedding, keep_prob1, keep_prob2)

    with tf.name_scope('loss'):
        valid_num = tf.cast(tf.reduce_sum(doc_len), dtype=tf.float32)
        loss_op = - tf.reduce_sum(y * tf.log(pred)) / valid_num + reg * FLAGS.l2_reg
        loss_assist_list = []

    with tf.name_scope('train'):
        optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.lr_main).minimize(loss_op)

    true_y_op = tf.argmax(y, 2, name="true_y_op")
    pred_y_op = tf.argmax(pred, 2, name="pred_y_op")

    print('build model done!\n')

    ########训练和验证过程#########
    # Training Code Block
    print_training_info()
    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True

    with tf.Session(config=tf_config) as sess:

        kf, fold, SID = KFold(n_splits=10), 1, 0 #十折交叉验证
        Id = []
        p_list, r_list, f1_list = [], [], []
        for train, test in kf.split(x_data):
            tr_x, tr_y, tr_sen_len, tr_doc_len, tr_word_dis = map(lambda x: x[train],
                [x_data, y_data, sen_len_data, doc_len_data, word_distance])
            te_x, te_y, te_sen_len, te_doc_len, te_word_dis = map(lambda x: x[test],
                [x_data, y_data, sen_len_data, doc_len_data, word_distance])
            precision_list, recall_list, FF1_list = [], [], []
            pre_list, true_list, pre_list_prob = [], [], []

            sess.run(tf.global_variables_initializer())
            print('############# fold {} ###############'.format(fold))
            fold += 1
            max_f1 = 0.0
            print('train docs: {}    test docs: {}'.format(len(tr_y), len(te_y)))

            '''*********Train********'''
            for epoch in range(FLAGS.training_iter):
                step = 1
                #train：feed_list = [x[index], y[index], sen_len[index], doc_len[index], word_dis[index], keep_prob1, keep_prob2]
                for train, _ in get_batch_data(tr_x, tr_y, tr_sen_len, tr_doc_len, tr_word_dis, FLAGS.keep_prob1, FLAGS.keep_prob2, FLAGS.batch_size):
                    _, loss, pred_y, true_y, pred_prob, doc_len_batch = sess.run(
                        [optimizer, loss_op, pred_y_op, true_y_op, pred, doc_len],
                        feed_dict=dict(zip(placeholders, train)))
                    acc, p, r, f1 = func.acc_prf(pred_y, true_y, doc_len_batch)
                    if step % 10 == 0:
                        print('epoch {}: step {}: loss {:.4f} acc {:.4f}'.format(epoch + 1, step, loss, acc))
                    step = step + 1
                # print("begin save!")
                # saver.save(sess, "./run_final/model.ckpt", global_step=step)

                '''*********Test********'''
                test = [te_x, te_y, te_sen_len, te_doc_len, te_word_dis, 1., 1.]
                loss, pred_y, true_y, pred_prob = sess.run(
                    [loss_op, pred_y_op, true_y_op, pred], feed_dict=dict(zip(placeholders, test)))

                end_time = time.time()

                true_list.append(true_y)
                pre_list.append(pred_y)
                pre_list_prob.append(pred_prob)

                #计算精确率准确率召回率和F值
                acc, p, r, f1 = func.acc_prf(pred_y, true_y, te_doc_len)
                precision_list.append(p)
                recall_list.append(r)
                FF1_list.append(f1)
                if f1 > max_f1:
                    max_acc, max_p, max_r, max_f1 = acc, p, r, f1
                print('\ntest: epoch {}: loss {:.4f} acc {:.4f}\np: {:.4f} r: {:.4f} f1: {:.4f} max_f1 {:.4f}\n'.format(
                    epoch + 1, loss, acc, p, r, f1, max_f1))

            Id.append(len(te_x))
            SID = np.sum(Id) - len(te_x)
            _, maxIndex = func.maxS(FF1_list)
            print("maxIndex:", maxIndex)
            print('Optimization Finished!\n')
            pred_prob = pre_list_prob[maxIndex]

            p_list.append(max_p)
            r_list.append(max_r)
            f1_list.append(max_f1)
        print("running time: ", str((end_time - start_time) / 60.))
        print_training_info()
        p, r, f1 = map(lambda x: np.array(x).mean(), [p_list, r_list, f1_list])
        print("f1_score in 10 fold: {}\naverage : {} {} {}\n".format(np.array(f1_list).reshape(-1, 1), round(p, 4), round(r, 4), round(f1, 4)))

        return p, r, f1

def print_training_info():
    print('\n\n>>>>>>>>>>>>>>>>>>>>TRAINING INFO:\n')
    print('batch-{}, learning_rate-{}, keep_prob1-{}, num_heads-{}, n_layers-{}'.format(
        FLAGS.batch_size, FLAGS.lr_main, FLAGS.keep_prob1, FLAGS.num_heads, FLAGS.n_layers))
    print('training_iter-{}, scope-{}\n'.format(FLAGS.training_iter, FLAGS.scope))


def get_batch_data(x, y, sen_len, doc_len, word_dis, keep_prob1, keep_prob2, batch_size, test=False):
    for index in func.batch_index(len(y), batch_size, test):
        feed_list = [x[index], y[index], sen_len[index], doc_len[index], word_dis[index], keep_prob1, keep_prob2]
        yield feed_list, len(index)


def senEncode_softmax(s_senEncode, w_varible, b_varible, n_feature, doc_len):
    s = tf.reshape(s_senEncode, [-1, n_feature])
    s = tf.nn.dropout(s, keep_prob=FLAGS.keep_prob2)
    w = func.get_weight_varible(w_varible, [n_feature, FLAGS.n_class])
    b = func.get_weight_varible(b_varible, [FLAGS.n_class])
    print("s:{}".format(s))
    print("w:{}".format(w))
    pred = tf.matmul(s, w) + b
    print("matmul(s, w):{}".format(pred))
    pred *= func.getmask(doc_len, FLAGS.max_doc_len, [-1, 1])
    pred = tf.nn.softmax(pred)
    pred = tf.reshape(pred, [-1, FLAGS.max_doc_len, FLAGS.n_class], name='pred')
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


if __name__ == '__main__':
    tf.app.run()

