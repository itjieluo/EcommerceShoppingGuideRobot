import re
import pickle
import random
import numpy as np
import os
import pandas as pd


def random_embedding(word2id, embedding_dim):
    """

    :param word2id：a type of dict,字映射到id的词典
    :param embedding_dim：a type of int,embedding的维度
    :return embedding_mat：a type of list,返回一个二维列表，大小为[字数,embedding_dim]

    例：
    word2id:
        {"我":0,"爱":1,"你":2}
    embedding_dim:5

    返回：
    embedding_mat:
        [[-0.12973758,  0.18019868,  0.20711688,  0.17926247,  0.11360762],
         [ 0.06935755,  0.01281571,  0.1248916 , -0.08218211, -0.22710923],
         [-0.20481614, -0.02795857,  0.13419691, -0.24348333,  0.04530862]])
    """
    embedding_mat = np.random.uniform(-0.25, 0.25, (len(word2id), embedding_dim))
    embedding_mat = np.float32(embedding_mat)
    return embedding_mat


def make_dict(data_path):
    '''

    :param data_path: 文件路径
    :return word2id,id2word 返回字到id的映射、id到字的映射
    '''

    excel = pd.read_excel(data_path, keep_default_na=False)
    centence_list = excel["数据"]

    all_char = []
    all_char.append('<UNK>')
    all_char.append('<PAD>')
    for centence in centence_list:
        for char in centence:
            if char not in all_char:
                all_char.append(char)
    word2id = {}
    id2word = {}
    for index, char in enumerate(all_char):
        word2id[char] = index
        id2word[index] = char
    return word2id, id2word


def data_util(data_path, word2id):
    '''

    :param data_path:a type of str,数据文件的路径
    :param tag2id:a type of dict,标签到id的映射
    :return all_list:a type of list,处理后的数据,
            数据形式类似：[[[wordid,wordid,wordid...],seq_length,label],
                        [[wordid,wordid,wordid...],seq_length,label],
                        [[wordid,wordid,wordid...],seq_length,label],
                        [[wordid,wordid,wordid...],seq_length,label],
                        ......]
    '''


    excel = pd.read_excel(data_path, keep_default_na=False)
    centence_list = excel["数据"]
    label_list = excel["意图标注"]

    # with open(data_path, "r", encoding="utf8") as f:
        # data = f.read()
    # rr = re.compile(r'[,，。、“”‘’－》《（）●：！;…？]/O')
    # sentences = rr.split(data)
    # # ["海/O  钓/O  比/O  赛/O  地/O  点/O  在/O...", "这/O  座/O  依/O  山/O  傍/O  水/O.... "]
    # sentences = list(filter(lambda x: x.strip(), sentences))
    #
    # sentences = list(map(lambda x: x.strip(), sentences))

    all_list = []
    for i in range(len(centence_list)):
        char2id = []
        seq_length = 0
        label = label_list[i]
        if len(centence_list[i]) > 50:
            continue
        for char in centence_list[i]:
            # word_list = i.split()
            # # ["海/O","钓/O","比/O","赛/O","地/O","点/O","在/O".....]
            # one_list = []
            # wordids = [word2id[char] for word in word_list]
            # # [12,33,123,33,14,26,77....]
            # tagids = [tag2id[word.split('/')[1]] for word in word_list]
            # # [1,1,2,3,3,1,1,1,4,5,5....]
            # one_list.append(wordids)
            # one_list.append(tagids)
            # one_list.append(len(wordids))
            # all_list.append(one_list)
            char2id.append(word2id[char])
        seq_length = len(char2id)
        all_list.append([char2id, seq_length, label])
    random.shuffle(all_list)
    return all_list

def data_util_input(centence_list, word2id):
    all_list = []
    for i in range(len(centence_list)):
        char2id = []
        seq_length = 0
        label = "1"
        if len(centence_list[i]) > 50:
            continue
        for char in centence_list[i]:
            # word_list = i.split()
            # # ["海/O","钓/O","比/O","赛/O","地/O","点/O","在/O".....]
            # one_list = []
            # wordids = [word2id[char] for word in word_list]
            # # [12,33,123,33,14,26,77....]
            # tagids = [tag2id[word.split('/')[1]] for word in word_list]
            # # [1,1,2,3,3,1,1,1,4,5,5....]
            # one_list.append(wordids)
            # one_list.append(tagids)
            # one_list.append(len(wordids))
            # all_list.append(one_list)
            # char2id.append(word2id[char])
            try:
                char2id.append(word2id[char])
            except:
                char2id.append(word2id["<UNK>"])
        seq_length = len(char2id)
        all_list.append([char2id, seq_length, label])
    random.shuffle(all_list)
    return all_list

def get_batch(data, batch_size, word2id, shuffle=False):
    """

    :param data:a type of list,处理后的数据
    :param batch_size:a type of int,每个批次包含数据的数目
    :param word2id:a type of dict,字到id的映射
    :param shuffle:a type of boolean,是否打乱
    :return:np.array(res_seq):按批次的数据序列,并且每个batch的时间长度是一样的
            类似：[[2,31,22,12,341,23....],
                  [2,31,22,12,341,23....],
                  [2,31,22,12,341,23....]
                  ......]
            res_labels:按批次的数据对应的one-hot标签,并且每个batch的时间长度是一样的,shape大概是
                       [batch_size,time_step,num_tags]
            sentence_legth:按批次数据的序列长度
    """
    # 乱序没有加
    if shuffle:
        random.shuffle(data)
    pad = word2id['<PAD>']
    for i in range(len(data) // batch_size):
        data_size = data[i * batch_size: (i + 1) * batch_size]
        seqs, labels, sentence_legth = [], [], []
        for s, l, s_l in data_size:
            seqs.append(s)
            sentence_legth.append(l)
            if s_l == 1:
                labels.append([1, 0])
            elif s_l == 2:
                labels.append([0, 1])
        # max_l = max(sentence_legth)
        max_l = 50

        res_seq = []
        for sent in seqs:
            sent_new = np.concatenate((sent, np.tile(pad, max_l - len(sent))), axis=0)  # 以pad的形式补充成等长的帧数
            res_seq.append(sent_new)

        yield np.array(res_seq), np.array(labels), sentence_legth


def save_pickle(file_path, *args):
    with open(file_path, 'wb') as f1:
        pickle.dump(args, f1)


if __name__ == '__main__':
    # step1
    data_path = './意图数据/意图数据汇总_增强1.xlsx'
    file_path = './意图数据/data_增强1.pk'
    word2id, id2word = make_dict(data_path)
    print(len(word2id))
    print(word2id)
    #
    save_pickle(file_path, word2id, id2word)
    #
    # # step2
    # data = data_util(data_path, word2id)
    # print(len(data))
    # print(data[0])
    # for res_seq, res_labels, sentence_legth in get_batch(data, 64, word2id, shuffle=False):
    #     print(res_seq.shape)
    #     print(res_labels.shape)
