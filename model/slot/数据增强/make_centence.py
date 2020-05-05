"""
数据增强
"""

import pandas as pd

path = "../槽位数据/数据增强.xlsx"
path_padlist = "../槽位数据/数据增强轮询模板.xlsx"

excel_pad = pd.read_excel(path_padlist, keep_default_na=False)
rencheng = excel_pad["人称"]
shangpin = excel_pad["商品"]
pishu = excel_pad["匹数"]
pinpai = excel_pad["品牌"]
fangjian = excel_pad["房间"]
xingrong = excel_pad["形容"]
wenhou = excel_pad["问候"]

pad_dic = {
    "人称": rencheng,
    "商品": shangpin,
    "匹数": pishu,
    "品牌": pinpai,
    "房间": fangjian,
    "形容": xingrong,
    "问候": wenhou,
}

template_list = [
    ["我要一个{}的{}{}", ["形容", "匹数", "商品"], [0, 1, 2]],
    ["我要买个{}，{}用，{}的就好", ["商品", "房间", "形容"], [0, 1, 2]],
    ["{}{}最新的有什么样的", ["品牌", "商品"], [0, 1]],
    ["推荐一个{}，不要{}的", ["商品", "品牌"], [0 ,1]],
    ["买{}{}", ["品牌", "商品"], [0, 1]],
    ["我要{}{}", ["品牌", "商品"], [0, 1]],
    ["给我来一个{}{}", ["品牌", "商品"], [0, 1]],
    ["{}{}", ["品牌","商品"], [0, 1]],
]

# a = "{}要买{}，给{}来个{}"
# b = ["人称", "商品", "人称", "商品"]
# c = [0,1,0,1]
def data_enhance(template, pad_list, repeat):
    # data = []
    # if len(pad_list) == 1:
    #     for i in pad_dic[pad_list[0]]:
    #         if i != "":
    #             data.append(template.format(i))
    # elif len(pad_list) == 2:
    #     for i in pad_dic[pad_list[0]]:
    #         if i != "":
    #             for j in pad_dic[pad_list[1]]:
    #                 if j != "":
    #                     data.append(template.format(i,j))
    # elif len(pad_list) == 3:
    #     for i in pad_dic[pad_list[0]]:
    #         if i != "":
    #             for j in pad_dic[pad_list[1]]:
    #                 if j != "":
    #                     for k in pad_dic[pad_list[2]]:
    #                         if k != "":
    #                             data.append(template.format(i,j,k))
    # elif len(pad_list) == 4:
    #     for i in pad_dic[pad_list[0]]:
    #         if i != "":
    #             for j in pad_dic[pad_list[1]]:
    #                 if j != "":
    #                     for k in pad_dic[pad_list[2]]:
    #                         if k != "":
    #                             for l in pad_dic[pad_list[3]]:
    #                                 if l != "":
    #                                     data.append(template.format(i,j,k,l))


    data = []
    label_pinpai = []
    label_pishu = []
    label_fangjian = []

    if len(set(repeat)) == 1:
        for i in pad_dic[pad_list[0]]:
            if i != "":
                if len(pad_list) == 1:
                    temp = template.format(i)
                if len(pad_list) == 2:
                    temp = template.format(i,i)
                if len(pad_list) == 3:
                    temp = template.format(i,i,i)
                if len(pad_list) == 4:
                    temp = template.format(i,i,i,i)
                data.append(temp)

    if len(set(repeat)) == 2:
        if len(pad_list) == 2:
            for i in pad_dic[pad_list[0]]:
                if i !="":
                    for j in pad_dic[pad_list[1]]:
                        if j != "":
                            temp = template.format(i, j)
                            data.append(temp)
                            try:
                                a = pad_list.index("品牌")
                                if a == 0:
                                    label_pinpai.append(i)
                                elif a == 1:
                                    label_pinpai.append(j)
                            except:
                                label_pinpai.append("")
                            try:
                                a = pad_list.index("匹数")
                                if a == 0:
                                    label_pishu.append(i)
                                elif a == 1:
                                    label_pishu.append(j)
                            except:
                                label_pishu.append("")
                            try:
                                a = pad_list.index("房间")
                                if a == 0:
                                    label_fangjian.append(i)
                                elif a == 1:
                                    label_fangjian.append(j)
                            except:
                                label_fangjian.append("")


        if len(pad_list) == 3:
            if repeat[0] == repeat[1]:
                for i in pad_dic[pad_list[0]]:
                    if i != "":
                        for j in pad_dic[pad_list[2]]:
                            if j != "":
                                temp = template.format(i, i, j)
                                data.append(temp)

            if repeat[0] == repeat[2]:
                for i in pad_dic[pad_list[0]]:
                    if i  != "":
                        for j in pad_dic[pad_list[1]]:
                            if j != "":
                                temp = template.format(i, j, i)
                                data.append(temp)
            if repeat[1] == repeat[2]:
                for i in pad_dic[pad_list[0]]:
                    if i != "":
                        for j in pad_dic[pad_list[1]]:
                            if j != "":
                                temp = template.format(i, j, j)
                                data.append(temp)
        if len(pad_list) == 4:
            if repeat[0] == repeat[1] == repeat[2]:
                for i in pad_dic[pad_list[0]]:
                    if i != "":
                        for j in pad_dic[pad_list[3]]:
                            if j != "":
                                temp = template.format(i, i, i, j)
                                data.append(temp)
            if repeat[0] == repeat[1] == repeat[3]:
                for i in pad_dic[pad_list[0]]:
                    if i != "":
                        for j in pad_dic[pad_list[2]]:
                            if j != "":
                                temp = template.format(i, i, j, i)
                                data.append(temp)
            if repeat[0] == repeat[2] == repeat[3]:
                for i in pad_dic[pad_list[0]]:
                    if i != "":
                        for j in pad_dic[pad_list[1]]:
                            if j != "":
                                temp = template.format(i, j, i, i)
                                data.append(temp)

            if repeat[0] == repeat[1] and repeat[2] == repeat[3]:
                for i in pad_dic[pad_list[0]]:
                    if i != "":
                        for j in pad_dic[pad_list[2]]:
                             if j != "":
                                temp = template.format(i, i, j, j)
                                data.append(temp)
            if repeat[0] == repeat[2] and repeat[1] == repeat[3]:
                for i in pad_dic[pad_list[0]]:
                    if i != "":
                        for j in pad_dic[pad_list[1]]:
                            if j != "":
                                temp = template.format(i, j, i, j)
                                data.append(temp)
            if repeat[0] == repeat[3] and repeat[1] == repeat[2]:
                for i in pad_dic[pad_list[0]]:
                    if i != "":
                        for j in pad_dic[pad_list[1]]:
                            if j != "":
                                temp = template.format(i, j, j, i)
                                data.append(temp)

    if len(set(repeat)) == 3:
        if len(pad_list) == 3:
            for i in pad_dic[pad_list[0]]:
                if i != "":
                    for j in pad_dic[pad_list[1]]:
                        if j != "":
                            for k in pad_dic[pad_list[2]]:
                                if k != "":
                                    temp = template.format(i, j, k)
                                    data.append(temp)
                                    try:
                                        a = pad_list.index("品牌")
                                        if a == 0:
                                            label_pinpai.append(i)
                                        elif a == 1:
                                            label_pinpai.append(j)
                                        elif a == 2:
                                            label_pinpai.append(k)
                                    except:
                                        label_pinpai.append("")
                                    try:
                                        a = pad_list.index("匹数")
                                        if a == 0:
                                            label_pishu.append(i)
                                        elif a == 1:
                                            label_pishu.append(j)
                                        elif a == 2:
                                            label_pishu.append(k)
                                    except:
                                        label_pishu.append("")
                                    try:
                                        a = pad_list.index("房间")
                                        if a == 0:
                                            label_fangjian.append(i)
                                        elif a == 1:
                                            label_fangjian.append(j)
                                        elif a == 2:
                                            label_fangjian.append(k)
                                    except:
                                        label_fangjian.append("")
        if len(pad_list) == 4:
            if repeat[0] == repeat[1]:
                for i in pad_dic[pad_list[0]]:
                    if i != "":
                        for j in pad_dic[pad_list[2]]:
                            if j != "":
                                for k in pad_dic[pad_list[3]]:
                                    if k != "":
                                        temp = template.format(i, i, j, k)
                                        data.append(temp)
            if repeat[0] == repeat[2]:
                for i in pad_dic[pad_list[0]]:
                    if i != "":
                        for j in pad_dic[pad_list[1]]:
                            if j != "":
                                for k in pad_dic[pad_list[3]]:
                                    if k != "":
                                        temp = template.format(i, j, i, k)
                                        data.append(temp)
            if repeat[0] == repeat[3]:
                for i in pad_dic[pad_list[0]]:
                    if i != "":
                        for j in pad_dic[pad_list[1]]:
                            if j != "":
                                for k in pad_dic[pad_list[2]]:
                                    if k != "":
                                        temp = template.format(i, j, k, i)
                                        data.append(temp)
            if repeat[1] == repeat[2]:
                for i in pad_dic[pad_list[0]]:
                    if i != "":
                        for j in pad_dic[pad_list[1]]:
                            if j != "":
                                for k in pad_dic[pad_list[3]]:
                                    if k != "":
                                        temp = template.format(i, j, j, k)
                                        data.append(temp)
            if repeat[1] == repeat[3]:
                for i in pad_dic[pad_list[0]]:
                    if i != "":
                        for j in pad_dic[pad_list[1]]:
                            if j != "":
                                for k in pad_dic[pad_list[2]]:
                                    if k != "":
                                        temp = template.format(i, j, k, j)
                                        data.append(temp)
            if repeat[2] == repeat[3]:
                for i in pad_dic[pad_list[0]]:
                    if i != "":
                        for j in pad_dic[pad_list[1]]:
                            if j != "":
                                for k in pad_dic[pad_list[2]]:
                                    if k != "":
                                        temp = template.format(i, j, k, k)
                                        data.append(temp)
    return data, label_pinpai, label_pishu, label_fangjian

# d = data_enhance(a, b, c)

data = []
label_pinpai = []
label_pishu = []
label_fangjian = []
for template in template_list:
    data_, label_pinpai_, label_pishu_, label_fangjian_ = data_enhance(template[0], template[1], template[2])
    data += data_
    label_pinpai += label_pinpai_
    label_pishu += label_pishu_
    label_fangjian += label_fangjian_


dic = {"数据":data, "品牌":label_pinpai, "匹数":label_pishu, "房间":label_fangjian}
data = pd.DataFrame(dic)
pd.DataFrame(data).to_excel(path)
