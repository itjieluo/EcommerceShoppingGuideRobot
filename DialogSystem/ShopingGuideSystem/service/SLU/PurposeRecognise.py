

class Purpose:
    """
    目的类：
    实现意图识别功能：
        1.识别query的购买商品意图
        2.识别query的补充槽位意图
    算法：
        初版字符串匹配算法，后期可以考虑更换成文本分类算法。
        在初始化方法中添加模型初始化并使用self.X变量进行存储，使用的时候跑模型即可
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
        print("Purpose is real")

    def Init(self):
        """
        初始化
        """
        self.product_vocabulary = self.GetProductVocabulary()
        self.purpose = [1, 0, 0, 0, 0, 0]
        self.ismult = 0
        self.slot_vocabulary_airconditioning,\
        self.slot_vocabulary_refrigerator,\
        self.slot_vocabulary_washingmachine,\
        self.slot_vocabulary_television,\
        self.slot_vocabulary_electriccooker = self.GetSlotVocabulary()


    def GetPurpose(self, query):
        """
        接收传来的query,进行购买商品意图识别
        :param
            query: 传来的query
        :return
            purpose: 意图数组[0，0，0，0，0，0]
        """
        self.purpose = self.TextMatch(query, 0)
        return self.purpose


    def IsMult(self, query, product):
        """
        接受传来的query,进行槽位填充意图识别
        :param
            query: 传来的query
        :param
            product: 当前对话正在购买的商品
        :return
            ismult: True or False是否为槽位填充意图
        """
        self.ismult = self.TextMatch(query, 1, product)
        return self.ismult

    def TextMatch(self, query, match_type, product=None):
        """
        文本匹配，对query进行字符串匹配
        :param
            query: 要进行匹配的字符串
        :param
            match_type: 匹配种类：1：购买商品，2：槽位补充
        :param
            product: 字符串，正在进行中的对话购买的商品
        :return
            匹配类型1返回意图数组，匹配类型2返回0/1
        """
        if match_type == 0:
            self.purpose = [1, 0, 0, 0, 0, 0]
            for i in range(len(self.product_vocabulary)):
                if self.product_vocabulary[i] in query:
                    self.purpose[i+1] = 1
                    self.purpose[0] = 0
            return self.purpose
        elif match_type == 1:
            if product == "空调":
                for i in self.slot_vocabulary_airconditioning:
                    if i in query:
                        return 1
                return 0
            elif product == "冰箱":
                for i in self.slot_vocabulary_refrigerator:
                    if i in query:
                        return 1
                return 0
            elif product == "洗衣机":
                for i in self.slot_vocabulary_washingmachine:
                    if i in query:
                        return 1
                return 0
            elif product == "电视":
                for i in self.slot_vocabulary_television:
                    if i in query:
                        return 1
                return 0
            elif product == "电饭煲":
                for i in self.slot_vocabulary_electriccooker:
                    if i in query:
                        return 1
                return 0

    def GetProductVocabulary(self):
        """
        读取商品类型词汇表
        :return
            商品列表
        """
        product = []
        with open("ShopingGuideSystem/service/SLU/Vocabulary/product.txt", "r") as f:
            for line in f:
                product.append(line.split()[0])
        return product

    def GetSlotVocabulary(self):
        """
        读取槽位词汇表
        :return
            每个商品类别的槽位词汇列表
        """
        airconditioning = []
        refrigerator = []
        washingmachine = []
        television = []
        electriccooker = []
        with open("ShopingGuideSystem/service/SLU/Vocabulary/空调.txt", "r") as f:
            for line in f:
                airconditioning.append(line.split()[0])
        with open("ShopingGuideSystem/service/SLU/Vocabulary/冰箱.txt", "r") as f:
            for line in f:
                refrigerator.append(line.split()[0])
        with open("ShopingGuideSystem/service/SLU/Vocabulary/洗衣机.txt", "r") as f:
            for line in f:
                washingmachine.append(line.split()[0])
        with open("ShopingGuideSystem/service/SLU/Vocabulary/电视.txt", "r") as f:
            for line in f:
                television.append(line.split()[0])
        with open("ShopingGuideSystem/service/SLU/Vocabulary/电饭煲.txt", "r") as f:
            for line in f:
                electriccooker.append(line.split()[0])
        return airconditioning, refrigerator, washingmachine, television, electriccooker


# a = Purpose()
# a.Init()
# b = a.GetPurpose("我要买电饭煲锤子")
# print(b)
# c = a.IsMult("我要一个海尔的", "空调")
# print(c)
# d = a.IsMult("我要一个", "空调")
# print(d)

