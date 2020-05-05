from ShopingGuideSystem.service.DPO.Recall import Recall

class ActStrategy:
    """
    策略类：
        根据当前的对话状态进行动作策略选择
    算法：
        初版要求必须至少有一个槽位填充上才进行数据库查询否则要求用户补充信息
        槽位信息数量>0时，有商品策略。槽位信息数量=0时，需要补充槽位策略。
    变量：
        self.strategy 策略
        self.slot_suggest_list_airconditioning 空调推荐询问槽位列表
        self.slot_suggest_list_refrigerator 冰箱推荐询问槽位列表
        self.slot_suggest_list_washingmachine 洗衣机推荐询问槽位列表
        self.slot_suggest_list_television 电视推荐询问槽位列表
        self.slot_suggest_list_electriccooker 电饭煲推荐询问槽位列表
        self.recall Recall类
    方法：
        Init()  初始化
        GetStrategy(product, slot)  对外接口
        GetNeedSlot(product, slot)  获取需要的槽位名称
        GetSuggestSlotVocabulary()  读取建议槽位字典
    """
    def __init__(self):
        print("Strategy is real")

    def Init(self):
        """
        初始化
        """
        self.strategy = [0, 0, {}]
        self.slot_suggest_list_airconditioning, \
        self.slot_suggest_list_refrigerator, \
        self.slot_suggest_list_washingmachine, \
        self.slot_suggest_list_television, \
        self.slot_suggest_list_electriccooker = self.GetSuggestSlotVocabulary()
        self.recall = Recall()
        self.recall.Init()

    def GetStrategy(self, product, slot):
        """
        获取动作策略结果，对外接口
        :param
            product: 当前正在进行购买的商品种类
        :param
            slot: 已经识别到的槽位信息
        :return:
            strategy: 策略[策略，推荐补充槽位词典，商品信息]
        """
        if len(slot.keys()) == 0:
            self.strategy[0] = 0
            if product == "空调":
                self.strategy[1] = self.GetNeedSlot(product, slot)
            elif product == "冰箱":
                self.strategy[1] = self.GetNeedSlot(product, slot)
            elif product == "洗衣机":
                self.strategy[1] = self.GetNeedSlot(product, slot)
            elif product == "电视":
                self.strategy[1] = self.GetNeedSlot(product, slot)
            elif product == "电饭煲":
                self.strategy[1] = self.GetNeedSlot(product, slot)
        else:
            self.strategy[0] = 1
            self.strategy[1] = 0
            self.strategy[2] = self.recall.GetProduct(product, slot)
            # self.strategy[2] = {"查询结果":"1"}
        return self.strategy

    def GetNeedSlot(self, product, slot):
        """
        获取需要补充的槽位，对应NLG模块回复
        :param
            product: 正在购买的商品
        :param
            slot: 已经识别出来的槽位
        :return
            needslot: 需要的槽位序号 int
        """
        if product == "空调":
            for i in range(len(self.slot_suggest_list_airconditioning)):
                if self.slot_suggest_list_airconditioning[i] not in slot.keys():
                    return i + 1
        elif product == "冰箱":
            for i in range(len(self.slot_suggest_list_refrigerator)):
                if self.slot_suggest_list_refrigerator[i] not in slot.keys():
                    return i + 1
        elif product == "洗衣机":
            for i in range(len(self.slot_suggest_list_washingmachine)):
                if self.slot_suggest_list_washingmachine[i] not in slot.keys():
                    return i + 1
        elif product == "电视":
            for i in range(len(self.slot_suggest_list_television)):
                if self.slot_suggest_list_television[i] not in slot.keys():
                    return i + 1
        elif product == "电饭煲":
            for i in range(len(self.slot_suggest_list_electriccooker)):
                if self.slot_suggest_list_electriccooker[i] not in slot.keys():
                    return i + 1
        return 0

    def GetSuggestSlotVocabulary(self):
        """
        读取槽位词汇表
        :return
            每个商品类别的推荐槽位词汇列表
        """
        airconditioning = []
        refrigerator = []
        washingmachine = []
        television = []
        electriccooker = []
        with open("ShopingGuideSystem/service/DPO/Vocabulary/空调.txt", "r") as f:
            for line in f:
                airconditioning.append(line.split()[0])
        with open("ShopingGuideSystem/service/DPO/Vocabulary/冰箱.txt", "r") as f:
            for line in f:
                refrigerator.append(line.split()[0])
        with open("ShopingGuideSystem/service/DPO/Vocabulary/洗衣机.txt", "r") as f:
            for line in f:
                washingmachine.append(line.split()[0])
        with open("ShopingGuideSystem/service/DPO/Vocabulary/电视.txt", "r") as f:
            for line in f:
                television.append(line.split()[0])
        with open("ShopingGuideSystem/service/DPO/Vocabulary/电饭煲.txt", "r") as f:
            for line in f:
                electriccooker.append(line.split()[0])
        return airconditioning, refrigerator, washingmachine, television, electriccooker

# a = ActStrategy()
# a.Init()
# b = a.GetStrategy("空调", {"品牌":"美的"})
# print(b)