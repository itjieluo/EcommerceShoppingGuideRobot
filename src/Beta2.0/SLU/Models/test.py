import tensorflow as tf
import pickle
from tensorflow.contrib.crf import viterbi_decode
import data_util_slot as data_util


file_path = './data_slot增强1.pk'

with open(file_path, 'rb') as f1:
    word2id, id2word, tag2id, id2tag = pickle.load(f1)


sess = tf.Session()
saver = tf.train.import_meta_graph('./Slot/model.ckpt-1201.meta')  # 加载模型结构
saver.restore(sess, tf.train.latest_checkpoint('./Slot/'))  # 只需要指定目录就可以恢复所有变量信息

# 获取placeholder变量
input_x = sess.graph.get_tensor_by_name('word_ids:0')
dropout = sess.graph.get_tensor_by_name('dropout:0')
seq_len = sess.graph.get_tensor_by_name('sequence_lengths:0')

logits = sess.graph.get_tensor_by_name('classification/dense/BiasAdd:0')
transition_params = sess.graph.get_tensor_by_name('loss/transitions:0')

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

while True:

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
    for res_seq, res_labels, sentence_legth in data_util.get_batch_test([[s, s, len(s)]], 1, word2id,
                                                                        tag2id):
        # print("+++++++++++++++++++")
        # print(res_seq)
        # print(res_labels)
        # print(sentence_legth)
        logits_, transition_params_ = sess.run([logits, transition_params],
                                               feed_dict={
                                                   input_x: res_seq,

                                                   seq_len: sentence_legth,
                                                   dropout: 1.
                                               })
        # 获取真实序列、标签长度。
        pred_list = make_mask_test(logits_, sentence_legth, True,
                                   transition_params_)
        # print(pred_list)
        input_seq = [x for x in vali_data]
        print("输入序列：", input_seq)
        target_names = [id2tag[i] for i in pred_list]
        print("预测序列：", target_names)
