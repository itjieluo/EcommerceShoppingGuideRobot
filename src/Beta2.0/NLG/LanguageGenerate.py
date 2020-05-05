

class NaturalLanguageGenerate:
    """
    语言生成类：
        根据策略和语言模板生成回复语言
    算法：

    变量：
        self.answer 生成的回复
        self.answer_list_airconditioning 空调回复模版
        self.answer_list_refrigerator 冰箱回复模版
        self.answer_list_washingmachine 洗衣机回复模版
        self.answer_list_television 电视回复模版
        self.answer_list_electriccooker 电饭煲回复模版
    方法：
        Init()  初始化
        GetAnswer(product, strategy)  对外接口
        GetAnswerListVocabulary()  读取回复模版
    """
    def __init__(self):
        print("NLG is real")

    def Init(self):
        """
        初始化
        """
        self.answer = ""
        self.answer_list_airconditioning, \
        self.answer_list_refrigerator, \
        self.answer_list_washingmachine, \
        self.answer_list_television, \
        self.answer_list_electriccooker = self.GetAnswerListVocabulary()

    def GetAnswer(self, product, strategy):
        """
        获取语言生成的回复，对外接口
        :param
            strategy: 策略列表[策略[0,1], 推荐槽位int, 商品信息{"商品名称":"..."...}]
        :return
            answer: 回复语句 字符串
        """
        if product == "空调":
            if strategy[0] == 0:
                self.answer = self.answer_list_airconditioning[strategy[1]]
            else:
                self.answer = self.answer_list_airconditioning[strategy[1]]
        elif product == "冰箱":
            if strategy[0] == 0:
                self.answer = self.answer_list_airconditioning[strategy[1]]
            else:
                self.answer = self.answer_list_airconditioning[strategy[1]]
        elif product == "洗衣机":
            if strategy[0] == 0:
                self.answer = self.answer_list_airconditioning[strategy[1]]
            else:
                self.answer = self.answer_list_airconditioning[strategy[1]]
        elif product == "电视":
            if strategy[0] == 0:
                self.answer = self.answer_list_airconditioning[strategy[1]]
            else:
                self.answer = self.answer_list_airconditioning[strategy[1]]
        elif product == "电饭煲":
            if strategy[0] == 0:
                self.answer = self.answer_list_airconditioning[strategy[1]]
            else:
                self.answer = self.answer_list_airconditioning[strategy[1]]
        return self.answer

    def GetAnswerListVocabulary(self):
        """
        读取回复模版列表
        :return
            每个商品类别的回复模板列表
        """
        airconditioning = []
        refrigerator = []
        washingmachine = []
        television = []
        electriccooker = []
        with open("./NLG/Vocabulary/空调.txt", "r") as f:
            for line in f:
                airconditioning.append(line.split()[0])
        with open("./NLG/Vocabulary/冰箱.txt", "r") as f:
            for line in f:
                refrigerator.append(line.split()[0])
        with open("./NLG/Vocabulary/洗衣机.txt", "r") as f:
            for line in f:
                washingmachine.append(line.split()[0])
        with open("./NLG/Vocabulary/电视.txt", "r") as f:
            for line in f:
                television.append(line.split()[0])
        with open("./NLG/Vocabulary/电饭煲.txt", "r") as f:
            for line in f:
                electriccooker.append(line.split()[0])
        return airconditioning, refrigerator, washingmachine, television, electriccooker

# a = NaturalLanguageGenerate()
# a.Init()
# b = a.GetAnswer("空调", [0, 1, {}])
# print(b)
# b = a.GetAnswer("空调", [1, 0, {}])
# print(b)