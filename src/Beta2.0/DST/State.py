

class State:
    """
    状态类：
        记录当前对话状态：当前正在购买的商品和已经识别出来的槽位
    算法：
        修改、添加
    变量：
        self.slot 槽位提取结果字典{"品牌"："美的"，"能效"："一级"...}
    方法：
        Init()  初始化
        GetState()  获取当前状态
        ChangeState()  改变当前状态
    """
    def __init__(self):
        print("State is real")

    def Init(self):
        """
        初始化
        """
        self.state = ["", {}]

    def GetState(self):
        """

        :return
            state: 状态字典[商品，槽位]
        """
        return self.state

    def ChangeState(self, product, slot):
        """
        改变当前对话状态
        :param
            product: 正在进行购买的商品
        :param
            slot: 已经提取出来的槽位字典
        """
        self.state[0] = product
        self.state[1] = slot


# a = State()
# a.Init()
# b = a.GetState()
# print(b)
# a.ChangeState("空调", {"品牌": "美的", "频率": "变频"})
# b = a.GetState()
# print(b)