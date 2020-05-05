import pickle
# 划分训练集和测试集
# 构造词典
# 词转化为id表示
# 保存转化为id后的数据


# 读入数据
def read_train_data(file_path):
    with open(file_path, 'rb') as f:
        train_data = pickle.load(f)
    return train_data


def make_word_bag(words_bag,data_source):
    for sentence in data_source:
        for words in sentence:
            if words not in words_bag:
                words_bag.append(words)


def map_int(data_list,word2id):
    data_int = []
    for data in data_list:
        source_int = [word2id.get(word, word2id['<UNK>'])for word in data[0][:-1]]
        target_int = [word2id.get(word, word2id['<UNK>'])for word in data[1][:-1]] + [word2id['<EOS>']]
        data_int.append((source_int, target_int))
    return data_int


# map the character and pronunciation to idx
def source2index(data):
    """Map each character in the word list and special symbols to idx(int)
    Args:
        data: a list contains the words.
    Returns:
        idx2word: a dict map idx to character .
        word2idx: a dict map character to idx .
    """
    special_words = ['<EOS>', '<GO>', '<PAD>', '<UNK>']
    set_words = list(set([c for line in data for c in line] + special_words))

    idx2word = {idx: word for idx, word in enumerate(set_words)}
    word2idx = {word: idx for idx, word in idx2word.items()}
    return idx2word, word2idx


if __name__ == '__main__':
    # 读取数据
    train_data, test_data, word2id, idx2word = read_train_data("question_answer.pk")
    print(len(train_data), train_data[:2])

    # 数据转化为id表示
    train_data_int = map_int(train_data, word2id)
    test_data_int = map_int(test_data, word2id)
    print(len(train_data_int))
