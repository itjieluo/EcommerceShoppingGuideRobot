# -*- coding: utf-8 -*-
# Author: luojie
# Date: 2019-11-3
# 目的：根据用户输入query进行纠错。基于用户词表，拼音相似度与编辑距离的查询纠错。
# 实现：用户输入多数为拼音输入，故而将汉字转为拼音，找到拼音的编辑距离（通过修改，插入，删除，替换）找到相似的拼音，
# 判断相似的拼音在拼音字典中是否存在，若存在，则查找对应的词，根据词频返回最大词频的词
import jieba
import json
from xpinyin import Pinyin
from pypinyin import *


pinyin_word_path = "pinyin2word.model"


class SpellCorrect():
    def __init__(self,train=False, ret_num=10):
        """
        :param corpus_path: 文本路径
        :param train: 是否根据文本，生成纠错词库，默认文本为《水浒传》
        :param ret_num: 返回可能的纠错结果数目
        """
        self.p = Pinyin()
        self.ret_num = ret_num
        self.pinyin_word, self.word_count = self.load_json(pinyin_word_path)  # 字典，pinyin:words
        # self.word_count = self.load_json(word_count_path)
        self.WORDS = self.pinyin_word.keys()

    def candidates(self, word):
        "Generate possible spelling corrections for word."
        # print("word:",word)
        # self.known([word])：如果拼音在字典中存在，则返回拼音，
        # self.edits1(word)：不存在，找到编辑距离相似的，判断是否存在
        # self.edits2(word) ：从编辑距离中产生的拼音再产生相似的拼音返回
        # word：词典中不存在，返回原始输入的拼音
        res = (self.known([word]) or self.known(self.edits1(word)) or self.known(self.edits2(word)) or [word])
        return res

    def known(self, words):
        "The subset of `words` that appear in the dictionary of WORDS."
        word_set = set(w for w in words if w in self.WORDS)
        print("know_set:",word_set)
        return word_set
        # for w in words:
        #     if w in self.WORDS:
        #         print(233,w)

    def edits1(self, word):
        "All edits that are one edit away from `word`."
        letters = 'abcdefghijklmnopqrstuvwxyz'
        splits = [(word[:i], word[i:]) for i in range(len(word) + 1)]
        # print("split:",splits)
        deletes = [L + R[1:] for L, R in splits if R]
        # print("delete:",deletes)
        transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R) > 1]
        # print("transposes:",transposes)
        replaces = [L + c + R[1:] for L, R in splits if R for c in letters]
        # print("replaces:",replaces)
        inserts = [L + c + R for L, R in splits for c in letters]
        # print("inserts:",inserts)
        # print("result:",set(deletes + transposes + replaces + inserts))
        return set(deletes + transposes + replaces + inserts)

    def edits2(self, word):
        "All edits that are two edits away from `word` ."
        # print(233333)
        e2 = (e2 for e1 in self.edits1(word) for e2 in self.edits1(e1))
        # print("e2:",e2)
        return e2

    # def load_json(self, json_file_path):
    #     with open(json_file_path, encoding='utf-8') as f:
    #         return json.loads(f.read(), encoding='utf-8')

    def load_json(self, model_path):
        f = open(model_path, 'r', encoding="utf-8")
        a = f.read()
        word_dict = eval(a)
        word_fre = {}
        for key, value in word_dict.items():
            # print(111,key,value)
            word_fre.update(value)
        f.close()
        return word_dict,word_fre

    def dump_json(self, content, json_file_path):
        with open(json_file_path, 'w', encoding='utf-8') as f:
            json.dump(content, f, ensure_ascii=False, indent=4)

    def correct(self, word):
        # word_pinyin = self.p.get_pinyin(word, splitter='')
        # word_pinyin = self.p.get_pinyin(word, splitter='')
        word_pinyin = ','.join(lazy_pinyin(word))  # 拼音
        # print("word_pinyin:",word_pinyin) # 获取到拼音
        candidate_pinyin = self.candidates(word_pinyin)
        # print("candidate_pinyin:",candidate_pinyin)
        ret_dic = {}
        words = []
        for pinyin in candidate_pinyin:
            words.extend(self.pinyin_word[pinyin])
        # print("words:",words)
        for word in words:
            ret_dic[word] = self.word_count.get(word, 0)
        sort_word = sorted(ret_dic.items(), key=lambda x: x[1], reverse=True)
        return [item[0] for item in sort_word[:self.ret_num]]


if __name__ == '__main__':
    spell_correct = SpellCorrect()
    # print(spell_correct.correct('宋江'))
    # print(spell_correct.correct('松江'))
    # print(spell_correct.correct('李奎'))
    # print(spell_correct.correct('吴宋'))
    # print(spell_correct.correct('送三连'))
    # print(spell_correct.correct('苹果'))
    sentence = "我咬买小蜜手记"
    seg = jieba.cut(sentence)
    words_list = [w for w in seg]
    print(words_list)
    # words_list = ["我咬", "买", "手记"]
    temp = []
    for word in words_list:
        temp.append(spell_correct.correct(word)[0])
    new_sentence = "".join(temp)
    print(new_sentence)


