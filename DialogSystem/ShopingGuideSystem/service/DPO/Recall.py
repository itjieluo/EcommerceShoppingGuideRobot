from elasticsearch import Elasticsearch


class Recall:
    """
    召回类：
        进行数据库搜索召回
    算法：
        按商品、槽位构建ES查询语句，进行查询
    变量：
        self.es  实例化ES类进行查询
        self.product  商品查询结果json

    方法：
        Init()  初始化
        GetProduct(product, slot)  功能1对外接口 获取查询商品的结果
        SearchExpressionGenerate(slot)  生成查询语句，只针对槽位，因为商品在type级别控制
    """
    def __init__(self):
        print("Recall is real")

    def Init(self):
        """
        初始化
        """
        self.es = Elasticsearch([{"host":"127.0.0.1","port":9200}])

    def GetProduct(self, product, slot):
        """
        进行召回
        :param
            product: 正在进行购买的商品
        :param
            slot: 槽位字典{"品牌"：["美的"，"格力"...],"能效"：["一级"，"二级" ...],...}
        :return
            product: 召回结果
        """
        product_en, body = self.SearchExpressionGenerate(product, slot)
        json_of_product_info = self.es.search(index = product_en, doc_type = product_en, body = body)
        list_of_product_info = json_of_product_info["hits"]["hits"]
        self.product = list_of_product_info
        return self.product

    def SearchExpressionGenerate(self, product, slot_dict):
        """
        生成ES查询语句
        :param
            product: 正在购买的商品类别
        :param
            slot_dict: 提取出得槽位字典
        :return:
        """
        body = {"query":{"bool":{"must":[]}},"size":1}
        if slot_dict.keys() == []:
            body = {"query":{"match_all":{}}}
            return body
        for key in slot_dict:
            if key == "频率":
                key_expression = {"match": {"变频/定频": slot_dict[key]}}
            elif key == "匹数":
                key_expression = {"match": {"产品匹数": slot_dict[key]}}
            else:
                key_expression = {"match":{key:slot_dict[key]}}
            body["query"]["bool"]["must"].append(key_expression)
        if product == "空调":
            product_en = "aircondition"
        elif product == "冰箱":
            product_en = "refrigerator"
        elif product == "洗衣机":
            product_en = "washingmachine"
        elif product == "电视":
            product_en = "television"
        elif product == "电饭煲":
            product_en = "electriccooker"
        return product_en, body

# a = Recall()
# a.Init()
# print(a.GetProduct("aircondition", {}))
# print(a.GetProduct("aircondition", {"品牌": "美的"}))
# print(a.GetProduct("aircondition", {"品牌": "美的", "变频/定频": "变频"}))


