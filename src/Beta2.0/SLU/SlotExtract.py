import os
from SLU.Models.SLUModelInterface import SLUModel

class Slot:
    """
    槽位类：
        实现槽位提取功能
    算法：
        初版字符串匹配算法，后期可以考虑更换成序列标注算法。
        在初始化方法中添加模型初始化并使用self.X变量进行存储，使用的时候跑模型即可
    变量：
        self.slot 槽位提取结果字典{"品牌"："美的"，"能效"："一级"...}
        self.slot_dict_airconditioning 槽位字典{"品牌"：["美的"，"格力"...],"能效"：["一级"，"二级" ...],...}
        self.slot_dict_refrigerator
        self.slot_dict_washingmachine
        self.slot_dict_television
        self.slot_dict_electriccooker

    方法：
        Init()  初始化
        GetSlot(query, product)  对外功能接口
        TextMatch(query, product)  文本匹配函数
        GetSlotDict()  读取槽位字典

    """
    def __init__(self):
        print("Slot is real")

    def Init(self):
        """
        初始化
        """
        self.slot = {}
        self.slot_dict_airconditioning,\
        self.slot_dict_refrigerator,\
        self.slot_dict_washingmachine,\
        self.slot_dict_television,\
        self.slot_dict_electriccooker = self.GetSlotDict()

    def GetSlot(self, query, product, model = None):
        """
        接收传来的商品类别、query，对query进行槽位提取
        :param
            query: 传来的query
        :param
            product: 当前对话正在购买的商品
        :return
            slot: {"槽位名":数值,"槽位名":数值}

        """

        if product == "空调":
            self.seq, self.seq_tag = model.RunSlotModel(query)
            self.slot = self.SeqTagAnalyse(self.seq, self.seq_tag)
            print(self.slot)
        else:
            self.slot = self.TextMatch(query, product)
        # self.slot = self.TextMatch(query, product)
        return self.slot

    def TextMatch(self, query, product):
        """
        文本匹配，对query进行字符串匹配
        :param
            query: 要进行匹配的字符串
        :param
            product: 字符串，正在进行中的对话购买的商品
        :return
            匹配类型1返回意图数组，匹配类型2返回0/1
        """
        slot = {}
        if product == "空调":
            for i in self.slot_dict_airconditioning.keys():
                for j in self.slot_dict_airconditioning[i]:
                    if j in query:
                        slot[i] = j
            return slot
        elif product == "冰箱":
            for i in self.slot_dict_refrigerator.keys():
                for j in self.slot_dict_refrigerator[i]:
                    if j in query:
                        slot[i] = j
            return slot
        elif product == "洗衣机":
            for i in self.slot_dict_washingmachine.keys():
                for j in self.slot_dict_washingmachine[i]:
                    if j in query:
                        slot[i] = j
            return slot
        elif product == "电视":
            for i in self.slot_dict_television.keys():
                for j in self.slot_dict_television[i]:
                    if j in query:
                        slot[i] = j
            return slot
        elif product == "电饭煲":
            for i in self.slot_dict_electriccooker.keys():
                for j in self.slot_dict_electriccooker[i]:
                    if j in query:
                        slot[i] = j
            return slot

    def GetSlotDict(self):
        """
        读取每类商品每个槽位词汇字典
        :return
            每类商品类别每个槽位词汇字典
        """
        airconditioning = {}
        refrigerator = {}
        washingmachine = {}
        television = {}
        electriccooker = {}
        for i in os.listdir("./SLU/Vocabulary/空调"):
            with open("./SLU/Vocabulary/空调/" + i) as f:
                airconditioning[i[:-4]] = []
                for line in f:
                    airconditioning[i[:-4]].append(line.split()[0])
        for i in os.listdir("./SLU/Vocabulary/冰箱"):
            with open("./SLU/Vocabulary/冰箱/" + i) as f:
                refrigerator[i[:-4]] = []
                for line in f:
                    refrigerator[i[:-4]].append(line.split()[0])
        for i in os.listdir("./SLU/Vocabulary/洗衣机"):
            with open("./SLU/Vocabulary/洗衣机/" + i) as f:
                washingmachine[i[:-4]] = []
                for line in f:
                    washingmachine[i[:-4]].append(line.split()[0])
        for i in os.listdir("./SLU/Vocabulary/电视"):
            with open("./SLU/Vocabulary/电视/" + i) as f:
                television[i[:-4]] = []
                for line in f:
                    television[i[:-4]].append(line.split()[0])
        for i in os.listdir("./SLU/Vocabulary/电饭煲"):
            with open("./SLU/Vocabulary/电饭煲/" + i) as f:
                electriccooker[i[:-4]] = []
                for line in f:
                    electriccooker[i[:-4]].append(line.split()[0])
        return airconditioning, refrigerator, washingmachine, television, electriccooker

    def SeqTagAnalyse(self, seq, seq_tag):
        slot_temp = {}
        seq_tag.append("EOF")
        seq_state = "S"
        slot_key = ""
        slot_value = ""
        for i in range(len(seq_tag)):
            if seq_state == "S":
                if seq_tag[i] == "O":
                    pass
                elif "_B" in seq_tag[i]:
                    seq_state = "B"
                    if seq_tag[i] == "Brand_B":
                        slot_key = "品牌"
                        slot_value += seq[i]
                    elif seq_tag[i] == "Price_B":
                        slot_key = "价格"
                        slot_value += seq[i]
                    elif seq_tag[i] == "Power_B":
                        slot_key = "产品匹数"
                        slot_value += seq[i]
                    elif seq_tag[i] == "Frequency_B":
                        slot_key = "频率"
                        slot_value += seq[i]
                    elif seq_tag[i] == "Room_B":
                        slot_key = "房间"
                        slot_value += seq[i]
                    elif seq_tag[i] == "Style_B":
                        slot_key = "款式"
                        slot_value += seq[i]
                    elif seq_tag[i] == "Energy_efficiency_B":
                        slot_key = "能效等级"
                        slot_value += seq[i]
            elif seq_state == "B":
                seq_state = "I"
                if seq_tag[i] == "Brand_I":
                    slot_value += seq[i]
                elif seq_tag[i] == "Price_I":
                    slot_value += seq[i]
                elif seq_tag[i] == "Power_I":
                    slot_value += seq[i]
                elif seq_tag[i] == "Frequency_I":
                    slot_value += seq[i]
                elif seq_tag[i] == "Room_I":
                    slot_value += seq[i]
                elif seq_tag[i] == "Style_I":
                    slot_value += seq[i]
                elif seq_tag[i] == "Energy_efficiency_I":
                    slot_value += seq[i]
            elif seq_state == "I":
                if seq_tag[i] == "O":
                    slot_temp[slot_key] = slot_value
                    slot_key = ""
                    slot_value = ""
                    seq_state = "O"
                elif seq_tag[i] == "EOF":
                    slot_temp[slot_key] = slot_value
                    slot_key = ""
                    slot_value = ""
                    seq_state = "S"
                elif seq_tag[i] == "Brand_I":
                    slot_value += seq[i]
                elif seq_tag[i] == "Price_I":
                    slot_value += seq[i]
                elif seq_tag[i] == "Power_I":
                    slot_value += seq[i]
                elif seq_tag[i] == "Frequency_I":
                    slot_value += seq[i]
                elif seq_tag[i] == "Room_I":
                    slot_value += seq[i]
                elif seq_tag[i] == "Style_I":
                    slot_value += seq[i]
                elif seq_tag[i] == "Energy_efficiency_I":
                    slot_value += seq[i]
            elif seq_state == "O":
                if seq_tag[i] == "O":
                    pass
                elif seq_tag[i] == "EOF":
                    slot_key = ""
                    slot_value = ""
                    seq_state = "S"
                elif "_B" in i:
                    seq_state = "B"
                    if i == "Brand_B":
                        slot_key = "品牌"
                        slot_value += seq[i]
                    elif i == "Price_B":
                        slot_key = "价格"
                        slot_value += seq[i]
                    elif i == "Power_B":
                        slot_key = "产品匹数"
                        slot_value += seq[i]
                    elif i == "Frequency_B":
                        slot_key = "频率"
                        slot_value += seq[i]
                    elif i == "Room_B":
                        slot_key = "房间"
                        slot_value += seq[i]
                    elif i == "Style_B":
                        slot_key = "款式"
                        slot_value += seq[i]
                    elif i == "Energy_efficiency_B":
                        slot_key = "能效等级"
                        slot_value += seq[i]
        return slot_temp
# a = Purpose()
# a.Init()
# b = a.GetPurpose("我要买空调")
# print(b)
# c = a.IsMult("我要一个美的的", "空调")
# print(c)
# d = a.IsMult("我要一个", "空调")
# print(d)
# a = Slot()
# a.Init()
# b = a.GetSlot("美的一级一匹的", "空调")
# print(b)