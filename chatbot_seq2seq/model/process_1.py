import pickle
import re
import zhconv

# 问题最短的长度
min_q_len = 1
#  问题最长的长度
max_q_len = 30
#  答案最短的长度
min_a_len = 2
# 答案最长的长度
max_a_len = 30


def good_line(line):
    """
    判断一句话是否是好的语料
    """
    if len(line) == 0:
        return False
    ch_count = 0
    for c in line:
        # 中文字符范围
        if '\u4e00' <= c <= '\u9fff':
            ch_count += 1
    if ch_count / float(len(line)) >= 0.7 and len(re.findall(r'[a-zA-Z0-9]', ''.join(line))) < 3 \
            and len(re.findall(r'[ˇˊˋˍεπのゞェーω]', ''.join(line))) < 3:
        return True
    return False


# 读取数据，分词，划分Q和A
def read_file(file_path):
    word = []
    data = []
    with open(file_path, "r", encoding="utf-8")as f:
        # print(111, len(f.readlines()))
        for line in f.readlines():
            # 正则化处理特殊字符
            line = regular(line)
            x_temp = []
            # 去掉短句
            x = line.strip()
            # source_seged = jieba.cut(x)
            # target_seged = jieba.cut(y)
            for i in x.strip():
                x_temp.append(i)
                if i not in word:
                    word.append(i)
            data.append(x_temp)
    return data, word


# print(zhconv.convert('我幹什麼不干你事。', 'zh-cn'))  # 我干什么不干你事。

def regular(sen):
    """
    句子规范化，主要是对原始语料的句子进行一些标点符号的统一
    :param sen:
    :return:
    """
    # 繁体转简体
    sen = zhconv.convert(sen, 'zh-cn')
    sen = zhconv.convert(sen, 'zh-cn')
    sen = sen.replace('<GO>', '')
    sen = sen.replace('<PAD>', '')
    sen = sen.replace('<EOS>', '')
    sen = sen.replace('<UNK>', '')
    sen = sen.replace('/', '')
    sen = re.sub(r'…{1,100}', '···', sen)
    sen = re.sub(r'\.{3,100}', '···', sen)
    sen = re.sub(r'···{2,100}', '···', sen)
    sen = re.sub(r',{1,100}', '，', sen)
    sen = re.sub(r'，{1,100}', '，', sen)
    sen = re.sub(r'\.{1,100}', '。', sen)
    sen = re.sub(r'。{1,100}', '。', sen)
    sen = re.sub(r'\?{1,100}', '？', sen)
    sen = re.sub(r'？{1,100}', '？', sen)
    sen = re.sub(r'!{1,100}', '！', sen)
    sen = re.sub(r'！{1,100}', '！', sen)
    sen = re.sub(r'~{1,100}', '～', sen)
    sen = re.sub(r'～{1,100}', '～', sen)
    sen = re.sub(r'０', '0', sen)
    sen = re.sub(r'３', '3', sen)
    sen = re.sub(r'\s{1,100}', '，', sen)
    sen = re.sub(r'[“”]{1,100}', '"', sen)  #中文引号不好处理
    sen = re.sub('[^\w\u4e00-\u9fff"。，？！～·]+', '', sen)
    sen = re.sub(r'[ˇˊˋˍεπのゞェーω]', '', sen)
    return sen


def source2index(data):
    special_words = ['<GO>', '<EOS>', '<PAD>', '<UNK>']
    set_words = list(set(special_words+[c for line in data for c in line]))
    idx2word = {idx: word for idx, word in enumerate(set_words)}
    word2idx = {word: idx for idx, word in idx2word.items()}
    return idx2word, word2idx


def extract_data(data_list):
    res = []
    for data in data_list:
        q = data[0]
        a = data[1]
        if min_q_len <= len(q) <= max_q_len and min_a_len <= len(a) <= max_a_len:
            if good_line(q) and good_line(a):
                res.append((q, a))
    return res


def make_word_bag(res_list):
    words_bag = []
    for sentence in res_list:
        data = sentence[0]+sentence[1]
        for words in data:
            if words not in words_bag:
                words_bag.append(words)
    return words_bag


# save data
def save_data(data, file_name):
    with open(file_name, 'wb') as f:
        pickle.dump(data, f)


if __name__ == '__main__':
    # a = [1,2,3]
    # b = [2,3,4]
    # c = list(zip(a,b))
    # print("c",c)
    # d,e = list(zip(*c))
    # print("d",list(d))
    Q_path = "dialog/Q"
    A_path = "dialog/A"
    Q_data,Q_word = read_file(Q_path)
    A_data,A_word = read_file(A_path)
    print(len(Q_data), len(Q_word))  # 454051
    print(len(A_data), len(A_word))  # 454051

    # 判断数据集是否为好的数据，去除不好的数据集
    data = list(zip(Q_data, A_data))
    res_data = extract_data(data)

    print("res", len(res_data))  # 230286
    print(res_data[:10])

    # 划分训练集和测试集
    # train_data = res_data[:20000]
    # test_data = res_data[:20000]

    # 构造词典映射
    # words_bag = make_word_bag(train_data)
    words_bag = list(set(Q_word+A_word))
    print(len(words_bag), words_bag[:10])  # 5731

    idx2word, word2id = source2index(words_bag)
    print(word2id)
    print(word2id["<GO>"])

    final_data = [res_data, word2id, idx2word]
    save_data(final_data, "question_answer.pk")
    # save_data(test_data, "test_data.pk")
    print("保存数据成功")

