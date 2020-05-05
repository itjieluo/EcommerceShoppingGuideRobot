import pandas as pd
import numpy as np


def read_data(data_path):
    excel = pd.read_excel(data_path, keep_default_na=False)
    centence_list = excel["数据"]
    label_brand = excel["品牌"]
    label_price = excel["价格"]
    label_power = excel["匹数"]
    label_frequency = excel["变频\定频"]
    label_room = excel["房间大小"]
    label_style = excel["款式"]
    label_energy_efficiency = excel["能效等级"]

    return centence_list, label_brand, label_price, label_power, label_frequency, label_room, label_style, label_energy_efficiency

text, brand, price, power, frequency, room, style, energy_efficiency = read_data("./槽位数据/槽位数据汇总.xlsx")


text = list(text)
brand = list(brand)
price = list(price)
power = list(power)
frequency = list(frequency)
room = list(room)
style = list(style)
energy_efficiency = list(energy_efficiency)

print(text)
a_split = []
for i in range(len(text)):
    a_split.append(list(text[i]))
label = []
for i in a_split:
    label_temp = []
    for j in i:
        label_temp.append("O")
    label.append(label_temp)
print(label)


brand_list = []
for i in brand:
    brand_list.append(str(i).split("/"))
price_list = []
for i in price:
    price_list.append(str(i).split("/"))
power_list = []
for i in power:
    power_list.append(str(i).split("/"))
frequency_list = []
for i in frequency:
    frequency_list.append(str(i).split("/"))
room_list = []
for i in room:
    room_list.append(str(i).split("/"))
style_list = []
for i in style:
    style_list.append(str(i).split("/"))
energy_efficiency_list = []
for i in energy_efficiency:
    energy_efficiency_list.append(str(i).split("/"))

centence_list_split = []
for i in range(len(text)):
    centence_list_split.append(list(text[i]))

for centence_p in range(len(text)):
    if brand_list[centence_p] != [""] :
        for brand in brand_list[centence_p]:
            temp = text[centence_p].find(brand)
            # print(temp)
            for i in range(len(brand)):
                if i == 0:
                    label[centence_p][temp + i] = "Brand_B"
                else:
                    label[centence_p][temp + i] = "Brand_I"
    if price_list[centence_p] != [""]:
        for price in price_list[centence_p]:
            temp = text[centence_p].find(price)
            # print(temp)
            for i in range(len(price)):
                if i == 0:
                    label[centence_p][temp + i] = "Price_B"
                else:
                    label[centence_p][temp + i] = "Price_I"
    if power_list[centence_p] != [""] :
        for power in power_list[centence_p]:
            temp = text[centence_p].find(power)
            # print(temp)
            for i in range(len(power)):
                if i == 0:
                    label[centence_p][temp + i] = "Power_B"
                else:
                    label[centence_p][temp + i] = "Power_I"
    if frequency_list[centence_p] != [""]:
        for frequency in frequency_list[centence_p]:
            temp = text[centence_p].find(frequency)
            # print(temp)
            for i in range(len(frequency)):
                if i == 0:
                    label[centence_p][temp + i] = "Frequency_B"
                else:
                    label[centence_p][temp + i] = "Frequency_I"
    if room_list[centence_p] != [""] :
        for room in room_list[centence_p]:
            temp = text[centence_p].find(room)
            # print(temp)
            for i in range(len(room)):
                if i == 0:
                    label[centence_p][temp + i] = "Room_B"
                else:
                    label[centence_p][temp + i] = "Room_I"
    if style_list[centence_p] != [""]:
        for style in style_list[centence_p]:
            temp = text[centence_p].find(style)
            # print(temp)
            for i in range(len(style)):
                if i == 0:
                    label[centence_p][temp + i] = "Style_B"
                else:
                    label[centence_p][temp + i] = "Style_I"
    if energy_efficiency_list[centence_p] != [""] :
        for energy_efficiency in energy_efficiency_list[centence_p]:
            temp = text[centence_p].find(energy_efficiency)
            # print(temp)
            for i in range(len(energy_efficiency)):
                if i == 0:
                    label[centence_p][temp + i] = "Energy_efficiency_B"
                else:
                    label[centence_p][temp + i] = "Energy_efficiency_I"


for i in range(len(text)):
    print(text[i])
    print(label[i])


# a = ["我想要一个美的空调","给我来一太格力中央空调","美的空调不怎么样，我先要格力的空调"]
# b = [["美的"],["格力"],["美的","格力"]]
# c = [[],["中央空调"],[]]
#
# d = "美的/格力"
# e = "美的"
#
# print(d.split("/"))
# print(e.split("/"))
#
# a_split = []
# for i in range(len(a)):
#     a_split.append(list(a[i]))
#
# label = []
# for i in a_split:
#     label_temp = []
#     for j in i:
#         label_temp.append("O")
#     label.append(label_temp)
# print(label)
#
# for centence_p in range(len(a)):
#     if b[centence_p] != [] :
#         for brand in b[centence_p]:
#             temp = a[centence_p].find(brand)
#             print(temp)
#             for i in range(len(brand)):
#                 if i == 0:
#                     label[centence_p][temp + i] = "B_B"
#                 else:
#                     label[centence_p][temp + i] = "B_I"
#     if c[centence_p] != []:
#         for style in c[centence_p]:
#             temp = a[centence_p].find(style)
#             print(temp)
#             for i in range(len(style)):
#                 if i == 0:
#                     label[centence_p][temp + i] = "S_B"
#                 else:
#                     label[centence_p][temp + i] = "S_I"
#
# print(label)











