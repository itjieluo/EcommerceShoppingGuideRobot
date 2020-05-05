import tensorflow as tf
import numpy as np
import pickle
import params
import data_util
from tensorflow.contrib.crf import crf_log_likelihood
from tensorflow.contrib.crf import viterbi_decode
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

data_path = './意图数据/意图数据汇总_增强1.xlsx'
with open('./意图数据/data_增强1.pk', 'rb') as f1:
    word2id, id2word = pickle.load(f1)
data = data_util.data_util(data_path, word2id)

train_set = data[:-880]
test_set = data[-880:]

# 随机初始化的embedding方式
embeddings = data_util.random_embedding(word2id, params.embedding_dim)

# graph = tf.Graph()
# with graph.as_default():
#     word_ids = tf.placeholder(tf.int32, shape=[None, None], name="word_ids")
#     labels = tf.placeholder(tf.int32, shape=[None, None], name="labels")
#     sequence_lengths = tf.placeholder(tf.int32, shape=[None, ], name="sequence_lengths")
#     dropout_pl = tf.placeholder(dtype=tf.float32, shape=[], name="dropout")
#
#     with tf.variable_scope("words"):
#         _word_embeddings = tf.Variable(embeddings,
#                                        dtype=tf.float32,
#                                        trainable=params.update_embedding,
#                                        name="_word_embeddings")
#         # word_embeddings的shape是[None, None,params.embedding_dim]
#         word_embeddings = tf.nn.embedding_lookup(params=_word_embeddings,
#                                                  ids=word_ids,
#                                                  name="word_embeddings")
#         word_embeddings = tf.nn.dropout(word_embeddings, dropout_pl)
#
#     with tf.variable_scope("fb-lstm"):
#         cell_fw = [params.RNN_Cell(params.hidden_size) for _ in range(params.cell_nums)]
#         cell_bw = [params.RNN_Cell(params.hidden_size) for _ in range(params.cell_nums)]
#         rnn_cell_fw = tf.nn.rnn_cell.MultiRNNCell(cell_fw)
#         rnn_cell_bw = tf.nn.rnn_cell.MultiRNNCell(cell_bw)
#         (output_fw_seq, output_bw_seq), states = tf.nn.bidirectional_dynamic_rnn(rnn_cell_fw, rnn_cell_bw,
#                                                                                  word_embeddings,
#                                                                                  sequence_length=sequence_lengths,
#                                                                                  dtype=tf.float32)
#         # output_fw_seq [None, None, params.hidden_size]
#         # output_bw_seq [None, None, params.hidden_size]
#         output = tf.concat([output_fw_seq, output_bw_seq], axis=-1)
#         # output的shape是[None, None, params.hidden_size*2]
#         output = tf.nn.dropout(output, dropout_pl)
#
#     with tf.variable_scope("classification"):
#         # logits的shape是[None, None, params.num_tags]
#         logits = tf.layers.dense(output, params.num_tags)
#
#     with tf.variable_scope("loss"):
#         log_likelihood, transition_params = crf_log_likelihood(inputs=logits,
#                                                                tag_indices=labels,
#                                                                sequence_lengths=sequence_lengths)
#         loss = -tf.reduce_mean(log_likelihood)
#
#     with tf.variable_scope("train_step"):
#         global_step = tf.Variable(0, name="global_step", trainable=False)
#         global_add = global_step.assign_add(1)
#         optim = tf.train.AdamOptimizer(learning_rate=params.lr)
#
#         grads_and_vars = optim.compute_gradients(loss)
#         # 对梯度gradients进行裁剪，保证在[-params.clip, params.clip]之间。
#         grads_and_vars_clip = [[tf.clip_by_value(g, -params.clip, params.clip), v] for g, v in grads_and_vars]
#         train_op = optim.apply_gradients(grads_and_vars_clip, global_step=global_step)
#
#
# # 获取真实序列、标签长度。
# def make_mask(logits_, labels_, sentence_legth, is_CRF=False, transition_params_=None):
#     pred_list = []
#     label_list = []
#     for log, lab, seq_len in zip(logits_, labels_, sentence_legth):
#         if is_CRF:
#             viterbi_seq, _ = viterbi_decode(log[:seq_len], transition_params_)
#         else:
#             viterbi_seq = log[:seq_len]
#         pred_list.extend(viterbi_seq)
#         label_list.extend(lab[:seq_len])
#     return label_list, pred_list
#
#
# with tf.Session(graph=graph) as sess:
#     if params.isTrain:
#         saver = tf.train.Saver(tf.global_variables())
#         try:
#             ckpt_path = tf.train.latest_checkpoint('./checkpoint_crf/')
#             saver.restore(sess, ckpt_path)
#         except Exception:
#             init = tf.global_variables_initializer()
#             sess.run(init)
#         for epoch in range(params.epoch_num):
#             for res_seq, res_labels, sentence_legth in data_util.get_batch(train_set, params.batch_size, word2id,
#                                                                            tag2id, shuffle=params.shuffle):
#                 _, l, global_nums = sess.run([train_op, loss, global_add], {
#                     word_ids: res_seq,
#                     labels: res_labels,
#                     sequence_lengths: sentence_legth,
#                     dropout_pl: params.dropout
#                 })
#                 if global_nums % 50 == 0:
#                     # saver.save(sess, './checkpoint_crf/model.ckpt', global_step=global_nums)
#                     logits_, transition_params_ = sess.run([logits, transition_params],
#                                                            feed_dict={
#                                                                word_ids: res_seq,
#                                                                labels: res_labels,
#                                                                sequence_lengths: sentence_legth,
#                                                                dropout_pl: params.dropout
#                                                            })
#                     # 获取真实序列、标签长度。
#                     label_list, pred_list = make_mask(logits_, res_labels, sentence_legth, True,
#                                                                        transition_params_)
#                     all_list = np.concatenate((label_list, pred_list), axis=0)
#                     all_list = np.unique(all_list)
#                     target_names = [id2tag[i] for i in all_list]
#                     acc = accuracy_score(label_list, pred_list)
#                     print(
#                         'epoch {}, global_step {}, loss: {:.4}, accuracy: {:.4}  '.format(epoch + 1, global_nums + 1,
#                                                                                           l, acc))
#                     print(classification_report(label_list, pred_list, target_names=target_names))
#                 if global_nums % 200 == 0:
#                     print('-----------------valudation---------------')
#                     res_seq, res_labels, sentence_legth = next(
#                         data_util.get_batch(test_set, params.batch_size, word2id, tag2id, shuffle=params.shuffle))
#                     l, logits_, transition_params_ = sess.run([loss, logits, transition_params],
#                                                               feed_dict={
#                                                                   word_ids: res_seq,
#                                                                   labels: res_labels,
#                                                                   sequence_lengths: sentence_legth,
#                                                                   dropout_pl: params.dropout
#                                                               })
#                     # 获取真实序列、标签长度。
#                     label_list, pred_list = make_mask(logits_, res_labels, sentence_legth, True,
#                                                                        transition_params_)
#                     all_list = np.concatenate((label_list, pred_list), axis=0)
#                     all_list = np.unique(all_list)
#                     target_names = [id2tag[i] for i in all_list]
#                     acc = accuracy_score(label_list, pred_list)
#                     print('valudation_accuracy: {:.4}  '.format(acc))
#                     print(classification_report(label_list, pred_list, target_names=target_names))
#                     print('-----------------valudation---------------')
#         print('-----------------test---------------')
#         res_seq, res_labels, sentence_legth = next(
#             data_util.get_batch(test_set, len(test_set), word2id, tag2id, shuffle=params.shuffle))
#         l, logits_, transition_params_ = sess.run([loss, logits, transition_params],
#                                                   feed_dict={
#                                                       word_ids: res_seq,
#                                                       labels: res_labels,
#                                                       sequence_lengths: sentence_legth,
#                                                       dropout_pl: params.dropout
#                                                   })
#         # 获取真实序列、标签长度。
#         label_list, pred_list = make_mask(logits_, res_labels, sentence_legth, True,
#                                                            transition_params_)
#         all_list = np.concatenate((label_list, pred_list), axis=0)
#         all_list = np.unique(all_list)
#         target_names = [id2tag[i] for i in all_list]
#         acc = accuracy_score(label_list, pred_list)
#         print('test_accuracy: {:.4}  '.format(acc))
#         print(classification_report(label_list, pred_list, target_names=target_names))
#         print('-----------------test---------------')


num_classes = 2
batch_size = 64
embedding_size = 200
learning_rate = 0.001
filter_size = [2, 3, 4, 5]
optimizer_type = 1
max_model_num = 5
num_fliter = 400
epoch = 20
isTrain = 0
max_seqlen = 50

# 词汇总数
vocab_size = len(word2id)

# 定义计算图
graph = tf.Graph()
with graph.as_default():
    # define the global step of the graph
    global_step = tf.train.create_global_step(graph)
    # 定义占位符
    input_x = tf.placeholder(tf.int64, [None, max_seqlen], name='input_x')
    input_y = tf.placeholder(tf.float64, [None, num_classes], name='input_y')
    print(input_x)
    print(input_y)
    dropout_keep_prob = tf.placeholder(dtype=tf.float64, name="dropout_keep_prob")
    print(dropout_keep_prob)
    with tf.variable_scope("embedding2", dtype=tf.float64):
        embedding2 = tf.get_variable('embedding', shape=[vocab_size, embedding_size],
                                    initializer=tf.truncated_normal_initializer(stddev=0.001),
                                    )

        embedded_wrods2 = tf.nn.embedding_lookup(embedding2, input_x)
        w_drop = tf.nn.dropout(embedded_wrods2, dropout_keep_prob)
    # 对输入数据input_扩维
    x_expand = tf.expand_dims(w_drop, -1)
    # TextCNN
    pooling_output = []
    for i, filter_size_j in enumerate(filter_size):
        with tf.name_scope("conv-maxpool-%s" % filter_size_j):
            conv1 = tf.layers.conv2d(x_expand, filters=num_fliter, kernel_size=[filter_size_j, embedding_size],
                                     activation=tf.nn.leaky_relu,
                                     kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
                                     bias_initializer=tf.truncated_normal_initializer(stddev=0.01)
                                     )
            # print(1111,x_expand.shape.tolist()[1])
            conv_pooled = tf.nn.max_pool(
                conv1,
                ksize=[1, max_seqlen - filter_size_j + 1, 1, 1],
                strides=[1, 1, 1, 1],
                padding='VALID',
                name="pool")
            # print("max_pooling:", conv_pooled.shape)
            pooling_output.append(conv_pooled)
    pool_concat = tf.concat(pooling_output, 3)
    # print(2333,"pool_concat", pool_concat.shape)
    pool_concat_flat = tf.reshape(pool_concat, shape=(-1, num_fliter * len(filter_size)))
    h_drop = tf.nn.dropout(pool_concat_flat, dropout_keep_prob)

    layer1 = tf.layers.dense(inputs=h_drop, units=512, activation=tf.nn.leaky_relu,
                             kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
                             bias_initializer=tf.truncated_normal_initializer(stddev=0.01)
                             )
    logits = tf.layers.dense(inputs=layer1, units=num_classes, activation=None,
                             kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
                             bias_initializer=tf.truncated_normal_initializer(stddev=0.01)
                             )
    # print("logists:",logits.shape)
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=input_y))
    optimizer_collection = {0: tf.train.GradientDescentOptimizer(learning_rate),
                            1: tf.train.AdamOptimizer(learning_rate),
                            2: tf.train.RMSPropOptimizer(learning_rate)}
    # Using the optimizer defined by optimizer_type
    optimizer = optimizer_collection[optimizer_type]
    # compute gradient
    gradients = optimizer.compute_gradients(loss)
    # apply gradient clipping to prevent gradient explosion
    capped_gradients = [(tf.clip_by_norm(grad, 5), var) for grad, var in gradients if grad is not None]
    # capped_gradients = [(tf.clip_by_value(grad, -0.1, 0.1), var) for grad, var in gradients if grad is not None]
    # update the
    opt = optimizer.apply_gradients(capped_gradients, global_step=global_step)
    # opt = tf.train.AdamOptimizer(L).minimize(loss)
    summary_loss = tf.summary.scalar('loss', loss)

    true = tf.argmax(input_y, axis=1)
    pred_softmax = tf.nn.softmax(logits)
    pred = tf.argmax(pred_softmax, axis=1)
    # tp_op, tn_op, fp_op, fn_op = tf_confusion_metrics(pred, true)
    correct_prediction = tf.equal(tf.cast(pred, tf.int64), true)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    summary_acc = tf.summary.scalar('acc', accuracy)

    print(pred)

# 运行计算图
with tf.Session(graph=graph) as sess:
    # define summary file writer
    # writer = tf.summary.FileWriter("./summary/visualization/", graph=graph)
    # merged = tf.summary.merge_all()
    saver = tf.train.Saver(max_to_keep=max_model_num)
    if isTrain == 1:
        init = tf.global_variables_initializer()
        sess.run(init)
        step = 0
        for epc in range(epoch):
            # lo = 0.0
            for res_seq, res_labels, sentence_legth in data_util.get_batch(train_set, batch_size , word2id, shuffle=params.shuffle):
                # print(res_seq.shape)
                # print(res_labels.shape)
                # print(res_labels)
                print(epc, step)
                print("----------")
                _, loss_, T, P, acc = sess.run([opt, loss, true, pred, accuracy],
                                               {input_x: res_seq,
                                                input_y: res_labels,
                                                dropout_keep_prob:0.6})
                if step % 10 == 0:
                    confusion = confusion_matrix(P, T)
                    # print(T,P)
                    print("训练confusion_matrix:")
                    print(confusion)
                    print("训练:", classification_report(T, P, target_names=["0","1"]))
                    # 相关系数：
                    coefxy = np.corrcoef(T, P)
                    print("训练相关系数矩阵：",coefxy)
                    print("训练Step:{:>5}\n\tloss:{:>5}\n\taccuracy2：{:>7.2%}.".format(step,loss_,acc))
                    print("############测试################")
                    # 验证数据
                    for x2, y2, seq_l2 in data_util.get_batch(test_set, 880,
                                                                                   word2id, shuffle=params.shuffle):
                        vali_acc, t,p = sess.run([accuracy, true,pred],
                                                                    {input_x: x2,
                                                                     input_y: y2,
                                                                     dropout_keep_prob: 1,
                                                                     })
                        vali_confusion = confusion_matrix(p, t)
                        print(classification_report(t, p, target_names=["0","1"]))
                        print("测试confusion_matrix:")
                        print(vali_confusion)
                        test_coefxy = np.corrcoef(t, p)
                        print("测试集相关系数矩阵：", test_coefxy)
                        print("验证Step:{:>5}\n\taccuracy2：{:>7.2%}".format(step,vali_acc))
                saver.save(sess, save_path="./880验证_增强1/model.ckpt", global_step=step)
                step += 1
        # save the model when finished
        saver.save(sess, save_path='./880验证_增强1/model.ckpt', global_step=step)
        print('Model Trained and Saved')
        # 测试集
    elif isTrain == 0:
        while True:
            # load model from folder
            checkpoint = tf.train.latest_checkpoint('./880验证_增强1/')
            saver.restore(sess, checkpoint)
            vali_data = input("输入评论数据：")
            s = data_util.data_util_input([vali_data], word2id)
            for res_seq, res_labels, sentence_legth in data_util.get_batch(s, 1 , word2id, shuffle=params.shuffle):
                # print(res_seq.shape)
                # print(res_labels.shape)
                # print(res_labels)
                pred_= sess.run([pred],{input_x: res_seq,
                                                dropout_keep_prob:1.})
                print(pred_)


            #
            # # 将词映射为id表示
            # data_id = [word2id.get(word, word2id['UNK']) for word in vali_data]
            # print(data_id)
            # # 将数据补齐为最大长度
            # if len(data_id) < max_seqlen:
            #     data_id = data_id + [word2id["PAD"]] * (max_seqlen - len(data_id))
            # else:
            #     data_id = data_id[:50]
            # data_id = np.array(data_id).reshape(-1, max_seqlen)
            # test_pred = sess.run([pred], {input_x: data_id,
            #                               dropout_keep_prob: 1,
            #                               })
            # print("预测：", test_pred)