import tensorflow as tf
import numpy as np
import pickle
import params
import data_util
from tensorflow.contrib.crf import crf_log_likelihood
from tensorflow.contrib.crf import viterbi_decode
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"


data_path = '槽位数据/槽位数据汇总_增强1.xlsx'
file_path = '槽位数据/data_增强1.pk'

with open(file_path, 'rb') as f1:
    word2id, id2word, tag2id, id2tag = pickle.load(f1)

text, brand, price, power, frequency, room, style, energy_efficiency = data_util.read_data(data_path)
text_list, tag_list = data_util.get_text_and_label_list(text, brand, price, power, frequency, room, style, energy_efficiency)

data = data_util.data_util(text_list, tag_list, word2id, tag2id)

train_set = data[:-64]
test_set = data[-64:]

# 随机初始化的embedding方式
embeddings = data_util.random_embedding(word2id, params.embedding_dim)

graph = tf.Graph()
with graph.as_default():
    word_ids = tf.placeholder(tf.int32, shape=[None, None], name="word_ids")
    labels = tf.placeholder(tf.int32, shape=[None, None], name="labels")
    sequence_lengths = tf.placeholder(tf.int32, shape=[None, ], name="sequence_lengths")
    dropout_pl = tf.placeholder(dtype=tf.float32, shape=[], name="dropout")

    with tf.variable_scope("words"):
        _word_embeddings = tf.Variable(embeddings,
                                       dtype=tf.float32,
                                       trainable=params.update_embedding,
                                       name="_word_embeddings")
        # word_embeddings的shape是[None, None,params.embedding_dim]
        word_embeddings = tf.nn.embedding_lookup(params=_word_embeddings,
                                                 ids=word_ids,
                                                 name="word_embeddings")
        word_embeddings = tf.nn.dropout(word_embeddings, dropout_pl)

    with tf.variable_scope("fb-lstm"):
        cell_fw = [params.RNN_Cell(params.hidden_size) for _ in range(params.cell_nums)]
        cell_bw = [params.RNN_Cell(params.hidden_size) for _ in range(params.cell_nums)]
        rnn_cell_fw = tf.nn.rnn_cell.MultiRNNCell(cell_fw)
        rnn_cell_bw = tf.nn.rnn_cell.MultiRNNCell(cell_bw)
        (output_fw_seq, output_bw_seq), states = tf.nn.bidirectional_dynamic_rnn(rnn_cell_fw, rnn_cell_bw,
                                                                                 word_embeddings,
                                                                                 sequence_length=sequence_lengths,
                                                                                 dtype=tf.float32)
        # output_fw_seq [None, None, params.hidden_size]
        # output_bw_seq [None, None, params.hidden_size]
        output = tf.concat([output_fw_seq, output_bw_seq], axis=-1)
        # output的shape是[None, None, params.hidden_size*2]
        output = tf.nn.dropout(output, dropout_pl)

    with tf.variable_scope("classification"):
        # logits的shape是[None, None, params.num_tags]
        logits = tf.layers.dense(output, params.num_tags)

        print(logits)

    with tf.variable_scope("loss"):
        log_likelihood, transition_params = crf_log_likelihood(inputs=logits,
                                                               tag_indices=labels,
                                                               sequence_lengths=sequence_lengths)
        print(transition_params)
        loss = -tf.reduce_mean(log_likelihood)

    with tf.variable_scope("train_step"):
        global_step = tf.Variable(0, name="global_step", trainable=False)
        global_add = global_step.assign_add(1)
        optim = tf.train.AdamOptimizer(learning_rate=params.lr)

        grads_and_vars = optim.compute_gradients(loss)
        # 对梯度gradients进行裁剪，保证在[-params.clip, params.clip]之间。
        grads_and_vars_clip = [[tf.clip_by_value(g, -params.clip, params.clip), v] for g, v in grads_and_vars]
        train_op = optim.apply_gradients(grads_and_vars_clip, global_step=global_step)


# 获取真实序列、标签长度。
def make_mask(logits_, labels_, sentence_legth, is_CRF=False, transition_params_=None):
    pred_list = []
    label_list = []
    for log, lab, seq_len in zip(logits_, labels_, sentence_legth):
        if is_CRF:
            viterbi_seq, _ = viterbi_decode(log[:seq_len], transition_params_)
        else:
            viterbi_seq = log[:seq_len]
        pred_list.extend(viterbi_seq)
        label_list.extend(lab[:seq_len])
    return label_list, pred_list

# 获取真实序列、标签长度。
def make_mask_test(logits_, sentence_legth, is_CRF=False, transition_params_=None):
    pred_list = []
    # print(logits_)
    # print(sentence_legth)
    for log, seq_len in zip(logits_, sentence_legth):
        if is_CRF:
            viterbi_seq, _ = viterbi_decode(log[:seq_len], transition_params_)
            # print(viterbi_seq)
        else:
            viterbi_seq = log[:seq_len]
        pred_list.extend(viterbi_seq)
    return pred_list


# with tf.Session(graph=graph) as sess:
#     saver = tf.train.Saver(max_to_keep=params.max_model_num)
#     if params.isTrain:
#         # saver = tf.train.Saver(tf.global_variables())
#         # try:
#         #     ckpt_path = tf.train.latest_checkpoint('./checkpoint_crf/')
#         #     saver.restore(sess, ckpt_path)
#         # except Exception:
#         #     init = tf.global_variables_initializer()
#         #     sess.run(init)
#         for epoch in range(params.epoch_num):
#             print("epoch",epoch)
#             for res_seq, res_labels, sentence_legth in data_util.get_batch(train_set, params.batch_size, word2id,
#                                                                            tag2id, shuffle=params.shuffle):
#                 print(res_seq[0])
#                 print(res_labels[0])
#                 print(sentence_legth[0])
#                 _, l, global_nums = sess.run([train_op, loss, global_add], {
#                     word_ids: res_seq,
#                     labels: res_labels,
#                     sequence_lengths: sentence_legth,
#                     dropout_pl: params.dropout
#                 })
#                 if global_nums % 10 == 0:
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
#
#                 if global_nums % 20 == 0:
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
#             saver.save(sess, save_path="./32验证/model.ckpt", global_step=global_nums)
#         saver.save(sess, save_path="./32验证/model.ckpt", global_step=global_nums)
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
#
#     elif params.isTrain == 0:
#         while True:
#
#             # load model from folder
#             # checkpoint = tf.train.latest_checkpoint('./32验证/')
#             # saver.restore(sess, checkpoint)
#             vali_data = input("输入评论数据：")
#             s = data_util.data_util_input(vali_data, word2id)
#             print("------------------")
#             print(s)
#             # for res_seq, res_labels, sentence_legth in data_util.get_batch(s, 1, word2id, shuffle=params.shuffle):
#             #     # print(res_seq.shape)
#             #
#             #     # print(res_labels.shape)
#             #
#             #     # print(res_labels)
#             #
#             #     pred_ = sess.run([pred], {input_x: res_seq,
#             #
#             #                               dropout_keep_prob: 1.})
#
#                 # print(pred_)
#             print([s, s, 1])
#             for res_seq, res_labels, sentence_legth in data_util.get_batch([[s, s ,1]], 1, word2id,
#                                                                            tag2id, shuffle=params.shuffle):
#                 print("+++++++++++++++++++")
#                 print(res_seq)
#                 print(res_labels)
#                 print(sentence_legth)
#
#                 # _, l, global_nums = sess.run([train_op, loss, global_add], {
#                 #     word_ids: res_seq,
#                 #     labels: res_labels,
#                 #     sequence_lengths: sentence_legth,
#                 #     dropout_pl: params.dropout
#                 # })
with tf.Session(graph=graph) as sess:
    if params.isTrain:
        saver = tf.train.Saver(tf.global_variables())
        try:
            ckpt_path = tf.train.latest_checkpoint('./checkpoint_crf/')
            saver.restore(sess, ckpt_path)
        except Exception:
            init = tf.global_variables_initializer()
            sess.run(init)
        for epoch in range(params.epoch_num):
            for res_seq, res_labels, sentence_legth in data_util.get_batch(train_set, params.batch_size, word2id,
                                                                           tag2id, shuffle=params.shuffle):
                _, l, global_nums = sess.run([train_op, loss, global_add], {
                    word_ids: res_seq,
                    labels: res_labels,
                    sequence_lengths: sentence_legth,
                    dropout_pl: params.dropout
                })
                if global_nums % 50 == 0:
                    # saver.save(sess, './checkpoint_crf/model.ckpt', global_step=global_nums)
                    logits_, transition_params_ = sess.run([logits, transition_params],
                                                           feed_dict={
                                                               word_ids: res_seq,
                                                               labels: res_labels,
                                                               sequence_lengths: sentence_legth,
                                                               dropout_pl: params.dropout
                                                           })
                    # 获取真实序列、标签长度。
                    label_list, pred_list = make_mask(logits_, res_labels, sentence_legth, True,
                                                                       transition_params_)
                    all_list = np.concatenate((label_list, pred_list), axis=0)
                    all_list = np.unique(all_list)
                    target_names = [id2tag[i] for i in all_list]
                    acc = accuracy_score(label_list, pred_list)
                    print(
                        'epoch {}, global_step {}, loss: {:.4}, accuracy: {:.4}  '.format(epoch + 1, global_nums + 1,
                                                                                          l, acc))
                    print(classification_report(label_list, pred_list, target_names=target_names))
                if global_nums % 200 == 0:
                    print('-----------------valudation---------------')
                    res_seq, res_labels, sentence_legth = next(
                        data_util.get_batch(test_set, params.batch_size, word2id, tag2id, shuffle=params.shuffle))
                    l, logits_, transition_params_ = sess.run([loss, logits, transition_params],
                                                              feed_dict={
                                                                  word_ids: res_seq,
                                                                  labels: res_labels,
                                                                  sequence_lengths: sentence_legth,
                                                                  dropout_pl: params.dropout
                                                              })
                    # 获取真实序列、标签长度。
                    label_list, pred_list = make_mask(logits_, res_labels, sentence_legth, True,
                                                                       transition_params_)
                    all_list = np.concatenate((label_list, pred_list), axis=0)
                    all_list = np.unique(all_list)
                    target_names = [id2tag[i] for i in all_list]
                    acc = accuracy_score(label_list, pred_list)
                    print('valudation_accuracy: {:.4}  '.format(acc))
                    print(classification_report(label_list, pred_list, target_names=target_names))
                    print('-----------------valudation---------------')
        saver.save(sess, save_path='./32验证/model.ckpt', global_step=global_nums+1)
        print('-----------------test---------------')
        res_seq, res_labels, sentence_legth = next(
            data_util.get_batch(test_set, len(test_set), word2id, tag2id, shuffle=params.shuffle))
        l, logits_, transition_params_ = sess.run([loss, logits, transition_params],
                                                  feed_dict={
                                                      word_ids: res_seq,
                                                      labels: res_labels,
                                                      sequence_lengths: sentence_legth,
                                                      dropout_pl: params.dropout
                                                  })
        # 获取真实序列、标签长度。
        label_list, pred_list = make_mask(logits_, res_labels, sentence_legth, True,
                                                           transition_params_)
        all_list = np.concatenate((label_list, pred_list), axis=0)
        all_list = np.unique(all_list)
        target_names = [id2tag[i] for i in all_list]
        acc = accuracy_score(label_list, pred_list)
        print('test_accuracy: {:.4}  '.format(acc))
        print(classification_report(label_list, pred_list, target_names=target_names))
        print('-----------------test---------------')
    elif params.isTrain == 0:
        while True:
            # load model from folder
            saver = tf.train.Saver(tf.global_variables())
            checkpoint = tf.train.latest_checkpoint('./32验证/')
            saver.restore(sess, checkpoint)
            vali_data = input("输入评论数据：")
            s = data_util.data_util_input(vali_data, word2id)
            # print("------------------")
            # print(vali_data)
            # for res_seq, res_labels, sentence_legth in data_util.get_batch(s, 1, word2id, shuffle=params.shuffle):
            #     # print(res_seq.shape)
            #
            #     # print(res_labels.shape)
            #
            #     # print(res_labels)
            #
            #     pred_ = sess.run([pred], {input_x: res_seq,
            #
            #                               dropout_keep_prob: 1.})

                # print(pred_)
            # print([s, s, len(s)])
            for res_seq, res_labels, sentence_legth in data_util.get_batch_test([[s, s ,len(s)]], 1, word2id,
                                                                           tag2id, shuffle=params.shuffle):
                # print("+++++++++++++++++++")
                # print(res_seq)
                # print(res_labels)
                # print(sentence_legth)
                logits_, transition_params_ = sess.run([logits, transition_params],
                                                          feed_dict={
                                                              word_ids: res_seq,
                                                              labels: res_labels,
                                                              sequence_lengths: sentence_legth,
                                                              dropout_pl: params.dropout
                                                          })
                # 获取真实序列、标签长度。
                pred_list = make_mask_test(logits_, sentence_legth, True,
                                                  transition_params_)
                # print(pred_list)
                input_seq = [x for x in vali_data]
                print("输入序列：", input_seq)
                target_names = [id2tag[i] for i in pred_list]
                print("预测序列：", target_names)

