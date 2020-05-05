from ShopingGuideSystem.service.SLU.PurposeRecognise import Purpose
from ShopingGuideSystem.service.SLU.SlotExtract import Slot
from ShopingGuideSystem.service.DPO.Strategy import ActStrategy
from ShopingGuideSystem.service.DST.State import State
from ShopingGuideSystem.service.NLG.LanguageGenerate import NaturalLanguageGenerate

class CentralControl:
    """
    总控类：
        实现总控功能
        启动对话系统，实例化、初始化各个子模块
        控制对话进程，调用各个子模块
    算法：

    变量：
        self.purpose 意图类
        self.slot 槽位类
        self.strategy 策略类
        self.state 状态类
        self.nlg 语言生成类
        self.answer 回复[回复类型, 回复文本, 商品信息]
        self.state_now 当前对话状态，用来接收state的返回值
        self.purpose_now 当前query意图，用来接收purpose返回值
        self.slot_now 当前提取出来的槽位，用来接收slot返回值
        self.strategy_now 当前策略，用来接收strategy返回值
        self.slot_new 新提取出得槽位，用来进行多轮时提取新query中的槽位
    方法：
        Init()  初始化
        Sess(query) 完成对话流程
    """
    def __init__(self):
        print("CentralControl is real")

    def Init(self):
        """
        初始化
        """
        self.purpose = Purpose()
        self.slot = Slot()
        self.strategy = ActStrategy()
        self.state = State()
        self.nlg = NaturalLanguageGenerate()

        self.purpose.Init()
        self.slot.Init()
        self.strategy.Init()
        self.state.Init()
        self.nlg.Init()

        self.answer = [0, "", {}]

    def Sess(self, query):
        """
        完成对话流程
        :param
            query: 传来的query
        :return
            answer: 回复列表[回复种类, 回复话语, 商品信息]
        """
        self.state_now = self.state.GetState() #获取对话状态
        if self.state_now[0] == "": #初始化状态时
            self.purpose_now = self.purpose.GetPurpose(query) #进行购买商品意图识别
            if self.purpose_now[0] == 1: #闲聊时
                self.answer[0] = 0
                self.answer[1] = self.nlg.GetAnswer(self.product_now, self.strategy_now)
            elif self.purpose_now[1] == 1: #购买空调时
                self.product_now = "空调"
                self.slot_now = self.slot.GetSlot(query, self.product_now) #进行槽位提取
                self.state.ChangeState(self.product_now, self.slot_now) #改变对话状态
                self.strategy_now = self.strategy.GetStrategy(self.product_now, self.slot_now) #获取回复策略
                if self.strategy_now[0] == 0: #槽位不足时
                    self.answer[0] = 0
                    self.answer[1] = self.nlg.GetAnswer(self.product_now, self.strategy_now)
                elif self.strategy_now[0] == 1: #槽位充足时
                    self.answer[0] = 1
                    self.answer[1] = self.nlg.GetAnswer(self.product_now, self.strategy_now)
                    self.answer[2] = self.strategy_now[2]
            elif self.purpose_now[2] == 1:
                self.product_now = "冰箱"
                self.slot_now = self.slot.GetSlot(query, self.product_now)
                self.state.ChangeState(self.product_now, self.slot_now)
                self.strategy_now = self.strategy.GetStrategy(self.product_now, self.slot_now)
                if self.strategy_now[0] == 0:
                    self.answer[0] = 0
                    self.answer[1] = self.nlg.GetAnswer(self.product_now, self.strategy_now)
                elif self.strategy_now[0] == 1:
                    self.answer[0] = 1
                    self.answer[1] = self.nlg.GetAnswer(self.product_now, self.strategy_now)
                    self.answer[2] = self.strategy_now[2]
            elif self.purpose_now[3] == 1:
                self.product_now = "洗衣机"
                self.slot_now = self.slot.GetSlot(query, self.product_now)
                self.state.ChangeState(self.product_now, self.slot_now)
                self.strategy_now = self.strategy.GetStrategy(self.product_now, self.slot_now)
                if self.strategy_now[0] == 0:
                    self.answer[0] = 0
                    self.answer[1] = self.nlg.GetAnswer(self.product_now, self.strategy_now)
                elif self.strategy_now[0] == 1:
                    self.answer[0] = 1
                    self.answer[1] = self.nlg.GetAnswer(self.product_now, self.strategy_now)
                    self.answer[2] = self.strategy_now[2]
            elif self.purpose_now[4] == 1:
                self.product_now = "电视"
                self.slot_now = self.slot.GetSlot(query, self.product_now)
                self.state.ChangeState(self.product_now, self.slot_now)
                self.strategy_now = self.strategy.GetStrategy(self.product_now, self.slot_now)
                if self.strategy_now[0] == 0:
                    self.answer[0] = 0
                    self.answer[1] = self.nlg.GetAnswer(self.product_now, self.strategy_now)
                elif self.strategy_now[0] == 1:
                    self.answer[0] = 1
                    self.answer[1] = self.nlg.GetAnswer(self.product_now, self.strategy_now)
                    self.answer[2] = self.strategy_now[2]
            elif self.purpose_now[5] == 1:
                self.product_now = "电饭煲"
                self.slot_now = self.slot.GetSlot(query, self.product_now)
                self.state.ChangeState(self.product_now, self.slot_now)
                self.strategy_now = self.strategy.GetStrategy(self.product_now, self.slot_now)
                if self.strategy_now[0] == 0:
                    self.answer[0] = 0
                    self.answer[1] = self.nlg.GetAnswer(self.product_now, self.strategy_now)
                elif self.strategy_now[0] == 1:
                    self.answer[0] = 1
                    self.answer[1] = self.nlg.GetAnswer(self.product_now, self.strategy_now)
                    self.answer[2] = self.strategy_now[2]
        else: #已经处于买某类商品时
            self.product_now = self.state_now[0] #获取当前正在购买的商品类别
            self.slot_now = self.state_now[1] #获取当前槽位信息
            self.purpose_now = self.purpose.IsMult(query, self.product_now) #对query进行多轮意图识别
            if self.purpose_now == 0: #如果不含多轮意图，重复初始化状态操作
                self.state.Init()
                self.answer = [0, "", {}]
                self.purpose_now = self.purpose.GetPurpose(query)  # 进行购买商品意图识别
                if self.purpose_now[0] == 1:  # 不购买商品时
                    self.answer[0] = 0
                    self.answer[1] = "对不起业务暂缓开通"
                    self.answer[2] = {}
                elif self.purpose_now[1] == 1:  # 购买空调时
                    self.product_now = "空调"
                    self.slot_now = self.slot.GetSlot(query, self.product_now)  # 进行槽位提取
                    self.state.ChangeState(self.product_now, self.slot_now)  # 改变对话状态
                    self.strategy_now = self.strategy.GetStrategy(self.product_now, self.slot_now)  # 获取回复策略
                    if self.strategy_now[0] == 0:  # 槽位不足时
                        self.answer[0] = 0
                        self.answer[1] = self.nlg.GetAnswer(self.product_now, self.strategy_now)
                    elif self.strategy_now[0] == 1:  # 槽位充足时
                        self.answer[0] = 1
                        self.answer[1] = self.nlg.GetAnswer(self.product_now, self.strategy_now)
                        self.answer[2] = self.strategy_now[2]
                elif self.purpose_now[2] == 1:
                    self.product_now = "冰箱"
                    self.slot_now = self.slot.GetSlot(query, self.product_now)
                    self.state.ChangeState(self.product_now, self.slot_now)
                    self.strategy_now = self.strategy.GetStrategy(self.product_now, self.slot_now)
                    if self.strategy_now[0] == 0:
                        self.answer[0] = 0
                        self.answer[1] = self.nlg.GetAnswer(self.product_now, self.strategy_now)
                    elif self.strategy_now[0] == 1:
                        self.answer[0] = 1
                        self.answer[1] = self.nlg.GetAnswer(self.product_now, self.strategy_now)
                        self.answer[2] = self.strategy_now[2]
                elif self.purpose_now[3] == 1:
                    self.product_now = "洗衣机"
                    self.slot_now = self.slot.GetSlot(query, self.product_now)
                    self.state.ChangeState(self.product_now, self.slot_now)
                    self.strategy_now = self.strategy.GetStrategy(self.product_now, self.slot_now)
                    if self.strategy_now[0] == 0:
                        self.answer[0] = 0
                        self.answer[1] = self.nlg.GetAnswer(self.product_now, self.strategy_now)
                    elif self.strategy_now[0] == 1:
                        self.answer[0] = 1
                        self.answer[1] = self.nlg.GetAnswer(self.product_now, self.strategy_now)
                        self.answer[2] = self.strategy_now[2]
                elif self.purpose_now[4] == 1:
                    self.product_now = "电视"
                    self.slot_now = self.slot.GetSlot(query, self.product_now)
                    self.state.ChangeState(self.product_now, self.slot_now)
                    self.strategy_now = self.strategy.GetStrategy(self.product_now, self.slot_now)
                    if self.strategy_now[0] == 0:
                        self.answer[0] = 0
                        self.answer[1] = self.nlg.GetAnswer(self.product_now, self.strategy_now)
                    elif self.strategy_now[0] == 1:
                        self.answer[0] = 1
                        self.answer[1] = self.nlg.GetAnswer(self.product_now, self.strategy_now)
                        self.answer[2] = self.strategy_now[2]
                elif self.purpose_now[5] == 1:
                    self.product_now = "电饭煲"
                    self.slot_now = self.slot.GetSlot(query, self.product_now)
                    self.state.ChangeState(self.product_now, self.slot_now)
                    self.strategy_now = self.strategy.GetStrategy(self.product_now, self.slot_now)
                    if self.strategy_now[0] == 0:
                        self.answer[0] = 0
                        self.answer[1] = self.nlg.GetAnswer(self.product_now, self.strategy_now)
                    elif self.strategy_now[0] == 1:
                        self.answer[0] = 1
                        self.answer[1] = self.nlg.GetAnswer(self.product_now, self.strategy_now)
                        self.answer[2] = self.strategy_now[2]
            elif self.purpose_now == 1: #多轮意图时
                self.slot_new = self.slot.GetSlot(query, self.product_now) #进行新槽位提取
                for i in self.slot_new.keys(): #将新槽位和旧槽位进行增、改
                    self.slot_now[i] = self.slot_new[i]
                self.state.ChangeState(self.product_now, self.slot_now) #改变对话状态
                self.strategy_now = self.strategy.GetStrategy(self.product_now, self.slot_now) #获取回复策略
                if self.strategy_now[0] == 0:
                    self.answer[0] = 0
                    self.answer[1] = self.nlg.GetAnswer(self.product_now, self.strategy_now)
                elif self.strategy_now[0] == 1:
                    self.answer[0] = 1
                    self.answer[1] = self.nlg.GetAnswer(self.product_now, self.strategy_now)
                    self.answer[2] = self.strategy_now[2]
        return self.answer

# a = CentralControl()
# a.Init()
# b = a.Sess("我要买空调")
# print(b)
# b = a.Sess("来一个美的的")
# print(b)
# b = a.Sess("要变频的")
# print(b)
# b = a.Sess("要1.5匹的")
# print(b)

