from elasticsearch import Elasticsearch

class ES:
    """
    ES类：
        Elasticsearch类，此类实现ES搜索、查询功能
    算法：

    变量：
        self.slot 槽位提取结果字典{"品牌"："美的"，"能效"："一级"...}
    方法：
        Init()  初始化
        GetState()  获取当前状态
        ChangeState()  改变当前状态
    """
    def __init__(self):
        print("ES is real")

    def Init(self):
        """
        初始化es
        """
        self.es = Elasticsearch([{"host":"127.0.0.1","port":9200}])

    def Search(self, slot):
        """
        查询函数，需要
        :return
            state: 状态字典[商品，槽位]
        """
        return self.state
