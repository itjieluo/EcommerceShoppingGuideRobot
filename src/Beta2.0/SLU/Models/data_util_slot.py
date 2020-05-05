import pandas as pd
import pickle
import random
import numpy as np

def read_data(data_path):
    excel = pd.read_excel(data_path, keep_default_na=False)
    centence_list = excel["数据"]
    label_brand = excel["品牌"]
    label_price = excel["价格"]
    label_power = excel["匹数"]
    label_frequency = excel["变频\定频"]
    label_room = excel["房间大小"]
    label_style = excel["款式"]
    label_energy_efficiency = excel["能效等级"]

    return centence_list, label_brand, label_price, label_power, label_frequency, label_room, label_style, label_energy_efficiency

def get_text_and_label_list(text, brand, price, power, frequency, room, style, energy_efficiency):
    text = list(text)
    brand = list(brand)
    price = list(price)
    power = list(power)
    frequency = list(frequency)
    room = list(room)
    style = list(style)
    energy_efficiency = list(energy_efficiency)

    print(text)
    a_split = []
    for i in range(len(text)):
        a_split.append(list(text[i]))
    label = []
    for i in a_split:
        label_temp = []
        for j in i:
            label_temp.append("O")
        label.append(label_temp)
    print(label)

    brand_list = []
    for i in brand:
        brand_list.append(str(i).split("/"))
    price_list = []
    for i in price:
        price_list.append(str(i).split("/"))
    power_list = []
    for i in power:
        power_list.append(str(i).split("/"))
    frequency_list = []
    for i in frequency:
        frequency_list.append(str(i).split("/"))
    room_list = []
    for i in room:
        room_list.append(str(i).split("/"))
    style_list = []
    for i in style:
        style_list.append(str(i).split("/"))
    energy_efficiency_list = []
    for i in energy_efficiency:
        energy_efficiency_list.append(str(i).split("/"))

    centence_list_split = []
    for i in range(len(text)):
        centence_list_split.append(list(text[i]))

    for centence_p in range(len(text)):
        if brand_list[centence_p] != [""]:
            for brand in brand_list[centence_p]:
                temp = text[centence_p].find(brand)
                # print(temp)
                for i in range(len(brand)):
                    if i == 0:
                        label[centence_p][temp + i] = "Brand_B"
                    else:
                        label[centence_p][temp + i] = "Brand_I"
        if price_list[centence_p] != [""]:
            for price in price_list[centence_p]:
                temp = text[centence_p].find(price)
                # print(temp)
                for i in range(len(price)):
                    if i == 0:
                        label[centence_p][temp + i] = "Price_B"
                    else:
                        label[centence_p][temp + i] = "Price_I"
        if power_list[centence_p] != [""]:
            for power in power_list[centence_p]:
                temp = text[centence_p].find(power)
                # print(temp)
                for i in range(len(power)):
                    if i == 0:
                        label[centence_p][temp + i] = "Power_B"
                    else:
                        label[centence_p][temp + i] = "Power_I"
        if frequency_list[centence_p] != [""]:
            for frequency in frequency_list[centence_p]:
                temp = text[centence_p].find(frequency)
                # print(temp)
                for i in range(len(frequency)):
                    if i == 0:
                        label[centence_p][temp + i] = "Frequency_B"
                    else:
                        label[centence_p][temp + i] = "Frequency_I"
        if room_list[centence_p] != [""]:
            for room in room_list[centence_p]:
                temp = text[centence_p].find(room)
                # print(temp)
                for i in range(len(room)):
                    if i == 0:
                        label[centence_p][temp + i] = "Room_B"
                    else:
                        label[centence_p][temp + i] = "Room_I"
        if style_list[centence_p] != [""]:
            for style in style_list[centence_p]:
                temp = text[centence_p].find(style)
                # print(temp)
                for i in range(len(style)):
                    if i == 0:
                        label[centence_p][temp + i] = "Style_B"
                    else:
                        label[centence_p][temp + i] = "Style_I"
        if energy_efficiency_list[centence_p] != [""]:
            for energy_efficiency in energy_efficiency_list[centence_p]:
                temp = text[centence_p].find(energy_efficiency)
                # print(temp)
                for i in range(len(energy_efficiency)):
                    if i == 0:
                        label[centence_p][temp + i] = "Energy_efficiency_B"
                    else:
                        label[centence_p][temp + i] = "Energy_efficiency_I"
    return a_split, label

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


def make_dict(text_list, tag_list):

    '''

    :param text_list: 文本数据列表
    :param tag_list: 标签列表
    :return word2id,id2word,tag2id,id2tag 返回字到id的映射、id到字的映射、标签到id的映射、id到标签的映射
    '''

    # 海/O  钓/O  比/O  赛/O  地/O  点/O  在/O.....
    # ["海/O","钓/O","比/O","赛/O","地/O","点/O","在/O".....]
    all_char = []
    all_tag = []
    all_char.append('<UNK>')
    all_char.append('<PAD>')
    all_tag.append('x')
    for centence in text_list:
        for char in centence:
            if char not in all_char:
                all_char.append(char)
    for tag_cen in tag_list:
        for tag in tag_cen:
            if tag not in all_tag:
                all_tag.append(tag)
    word2id = {}
    id2word = {}
    tag2id = {}
    id2tag = {}
    for index, char in enumerate(all_char):
        word2id[char] = index
        id2word[index] = char
    for index, tag in enumerate(all_tag):
        tag2id[tag] = index
        id2tag[index] = tag
    return word2id, id2word, tag2id, id2tag


def data_util(text_list, tag_list, word2id, tag2id):

    '''

    :param text_list: 文本列表
    :param tag_list: 标签列表
    :param word2id:a type of dict,字到id的映射
    :param tag2id:a type of dict,标签到id的映射
    :return all_list:a type of list,处理后的数据,
            数据形式类似：[[[wordid,wordid,wordid...],[tagid,tagid,tagid......],seq_length],
                        [[wordid,wordid,wordid...],[tagid,tagid,tagid......],seq_length],
                        [[wordid,wordid,wordid...],[tagid,tagid,tagid......],seq_length],
                        [[wordid,wordid,wordid...],[tagid,tagid,tagid......],seq_length]
                        ......]
    '''

    all_list = []
    for i in range(len(text_list)):
        one_list = []
        wordids = [word2id[word] for word in text_list[i]]
        # [12,33,123,33,14,26,77....]
        tagids = [tag2id[tag] for tag in tag_list[i]]
        # [1,1,2,3,3,1,1,1,4,5,5....]
        one_list.append(wordids)
        one_list.append(tagids)
        one_list.append(len(wordids))
        all_list.append(one_list)
    random.shuffle(all_list)
    return all_list

def data_util_input(text_input, word2id):
    wordids = []
    for word in text_input:
        try:
            wordids.append(word2id[word])
        except:
            wordids.append(word2id["<UNK>"])
    # [12,33,123,33,14,26,77....]
    return wordids


def get_batch(data, batch_size, word2id, tag2id, shuffle=False):
    """

    :param data:a type of list,处理后的数据
    :param batch_size:a type of int,每个批次包含数据的数目
    :param word2id:a type of dict,字到id的映射
    :param tag2id:a type of id,字到id的映射
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
    tag_pad = tag2id["x"]
    for i in range(len(data) // batch_size):
        data_size = data[i * batch_size: (i + 1) * batch_size]
        # print(data_size)
        seqs, labels, sentence_legth = [], [], []
        for s, l, s_l in data_size:
            seqs.append(s)
            labels.append(l)
            sentence_legth.append(s_l)
        max_l = max(sentence_legth)

        res_seq = []
        for sent in seqs:
            sent_new = np.concatenate((sent, np.tile(pad, max_l - len(sent))), axis=0)  # 以pad的形式补充成等长的帧数
            res_seq.append(sent_new)
        res_labels = []
        for label in labels:
            label_new = np.concatenate((label, np.tile(tag_pad, max_l - len(label))), axis=0)  # 以pad的形式补充成等长的帧数
            res_labels.append(label_new)

        yield np.array(res_seq), np.array(res_labels), sentence_legth

def get_batch_test(data, batch_size, word2id, tag2id, shuffle=False):

    """

    :param data:a type of list,处理后的数据
    :param batch_size:a type of int,每个批次包含数据的数目
    :param word2id:a type of dict,字到id的映射
    :param tag2id:a type of id,字到id的映射
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
    # if shuffle:
    #     random.shuffle(data)
    # pad = word2id['<PAD>']
    # tag_pad = tag2id["x"]
    for i in range(len(data) // batch_size):
        data_size = data[i * batch_size: (i + 1) * batch_size]
        # print(data_size)
        seqs, labels, sentence_legth = [], [], []
        for s, l, s_l in data_size:
            seqs.append(s)
            labels.append(l)
            sentence_legth.append(s_l)
        # max_l = max(sentence_legth)

        res_seq = seqs
        # for sent in seqs:
        #     sent_new = np.concatenate((sent, np.tile(pad, max_l - len(sent))), axis=0)  # 以pad的形式补充成等长的帧数
        #     res_seq.append(sent_new)

        res_labels = labels
        # for label in labels:
        #     label_new = np.concatenate((label, np.tile(tag_pad, max_l - len(label))), axis=0)  # 以pad的形式补充成等长的帧数
        #     res_labels.append(label_new)

        yield np.array(res_seq), np.array(res_labels), sentence_legth


def save_pickle(file_path, *args):
    with open(file_path, 'wb') as f1:
        pickle.dump(args, f1)


if __name__ == '__main__':
    # step1
    data_path = '槽位数据/槽位数据汇总_增强1.xlsx'
    file_path = '槽位数据/data_增强1.pk'
    text, brand, price, power, frequency, room, style, energy_efficiency = read_data(data_path)
    text_list, tag_list = get_text_and_label_list(text, brand, price, power, frequency, room, style, energy_efficiency)


    word2id, id2word, tag2id, id2tag = make_dict(text_list, tag_list)
    print(len(word2id))
    print(word2id)
    print(len(tag2id))
    print(tag2id)
    save_pickle(file_path, word2id, id2word, tag2id, id2tag)

    # step2
    # data = data_util(text_list, tag_list, word2id, tag2id)
    # print(len(data))
    # print(data[0])
    # for res_seq, res_labels, sentence_legth in get_batch(data, 32, word2id, tag2id, shuffle=False):
    #     print(res_seq[0])
    #     print(res_labels[0])
    #     print(res_seq.shape)
    #     print(res_labels.shape)
