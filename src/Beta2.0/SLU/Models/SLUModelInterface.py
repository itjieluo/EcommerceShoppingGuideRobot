import os
import tensorflow as tf
import pickle
from tensorflow.contrib.crf import viterbi_decode
import SLU.Models.data_util_purpose as data_util_p
import SLU.Models.data_util_slot as data_util_s


class SLUModel:
    """
    语义理解模型类：
    实现离线模型加载功能：
        1.识别意图识别模型
        2.识别槽位提取模型
    变量：
        self.product_vocabulary  商品种类列表 ["空调"，"冰箱"...]
        self.purpose  意图数组 [1, 0, 0, 0, 0, 0]
        self.ismult  是否是填充槽位意图 1 or 0
        self.slot_vocabulary_airconditioning  空调槽位词典列表["美的"，"一级"...]
        self.slot_vocabulary_refrigerator  冰箱槽位词典
        self.slot_vocabulary_washingmachine  洗衣机槽位词典
        self.slot_vocabulary_television  电视槽位词典
        self.slot_vocabulary_electriccooker  电饭煲槽位词典
    方法：
        Init()  初始化
        GetPurpose(query)  功能1对外接口
        IsMult(query, product)  功能2对外接口
        TextMatch(query, match_type, product=None)  文本匹配函数
        GetProductVocabulary()  读取商品词典
        GetSlotVocabulary()  读取槽位词典
    """
    def __init__(self):
        print("SLUModel is real")

    def Init(self):
        try:
            os.path.exists('./Models/Purpose')
            print("Foundd Purpose model.")
        except:
            print("Not found Purpose model derectory.")
        try:
            os.path.exists('./Models/Slot')
            print("Foundd Slot model.")
        except:
            print("Not found SLot model derectory.")

    def InitModel(self):

        with open('./SLU/Models/data_purpose增强1.pk', 'rb') as f1:
            self.purpose_word2id, self.purpose_id2word = pickle.load(f1)
        self.purpose_graph = tf.Graph()
        with self.purpose_graph.as_default():
            self.purpose_sess = tf.Session(graph=self.purpose_graph)
            self.purpose_saver = tf.train.import_meta_graph('./SLU/Models/Purpose/model.ckpt-2960.meta')  # 加载模型结构
            self.purpose_saver.restore(self.purpose_sess, tf.train.latest_checkpoint('./SLU/Models/Purpose/'))  # 只需要指定目录就可以恢复所有变量信息

            # 获取placeholder变量
            self.purpose_input_x = self.purpose_sess.graph.get_tensor_by_name('input_x:0')
            self.purpose_dropout = self.purpose_sess.graph.get_tensor_by_name('dropout_keep_prob:0')
            self.purpose_pred = self.purpose_sess.graph.get_tensor_by_name('ArgMax_1:0')

        with open("./SLU/Models/data_slot增强1.pk", 'rb') as f1:
            self.slot_word2id, self.slot_id2word, self.slot_tag2id, self.slot_id2tag = pickle.load(f1)
        self.slot_graph = tf.Graph()
        with self.slot_graph.as_default():
            self.slot_sess = tf.Session(graph=self.slot_graph)
            self.slot_saver = tf.train.import_meta_graph('./SLU/Models/Slot/model.ckpt-1201.meta')  # 加载模型结构
            self.slot_saver.restore(self.slot_sess, tf.train.latest_checkpoint('./SLU/Models/Slot/'))  # 只需要指定目录就可以恢复所有变量信息

            # 获取placeholder变量
            self.slot_input_x = self.slot_sess.graph.get_tensor_by_name('word_ids:0')
            self.slot_dropout = self.slot_sess.graph.get_tensor_by_name('dropout:0')
            self.slot_seq_len = self.slot_sess.graph.get_tensor_by_name('sequence_lengths:0')

            self.slot_logits = self.slot_sess.graph.get_tensor_by_name('classification/dense/BiasAdd:0')
            self.slot_transition_params = self.slot_sess.graph.get_tensor_by_name('loss/transitions:0')


    def RunPursoleModel(self, input_sentence):
        s = data_util_p.data_util_input([input_sentence], self.purpose_word2id)
        for res_seq, res_labels, sentence_legth in data_util_p.get_batch(s, 1, self.purpose_word2id):
            res = self.purpose_sess.run(self.purpose_pred, {self.purpose_input_x: res_seq, self.purpose_dropout: 1.})
        return res

    def RunSlotModel(self, input_sentence):

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

        s = data_util_s.data_util_input(input_sentence, self.slot_word2id)

        for res_seq, res_labels, sentence_legth in data_util_s.get_batch_test([[s, s, len(s)]], 1, self.slot_word2id,
                                                                              self.slot_tag2id):

            logits_, transition_params_ = self.slot_sess.run([self.slot_logits, self.slot_transition_params],
                                                   feed_dict={
                                                       self.slot_input_x: res_seq,

                                                       self.slot_seq_len: sentence_legth,
                                                       self.slot_dropout: 1.
                                                   })
            # 获取真实序列、标签长度。
            pred_list = make_mask_test(logits_, sentence_legth, True,
                                       transition_params_)
            input_seq = [x for x in input_sentence]
            print("输入序列：", input_seq)
            target_names = [self.slot_id2tag[i] for i in pred_list]
            print("预测序列：", target_names)
        return input_seq, target_names


# a = SLUModel()
# a.Init()
# a.InitModel()
#
# b = a.RunPursoleModel("我要买空调")
# print(b)
#
#
# b = a.RunSlotModel("我要买格力空调")
# print(b)
#
# b = a.RunPursoleModel("我要买蝙蝠")
# print(b)
#
# b = a.RunSlotModel("我要买2000块左右的空调")
# print(b)
#
# b = a.RunPursoleModel("你好")
# print(b)
#
# b = a.RunSlotModel("卧室18平")
# print(b)
#
# b = a.RunPursoleModel("你他妈了个锤子")
# print(b)
#
# b = a.RunSlotModel("我要变频的")
# print(b)


