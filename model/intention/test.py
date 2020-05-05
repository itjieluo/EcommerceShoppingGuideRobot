# import os
# print(os.listdir("./意图数据"))
#
# with open("意图数据/zhidao.txt", "r") as f:
#     data = []
#     for line in f:
#         print(line.split("\t"))
#         for i in range(2):
#             a = line.split("\t")[i]
#             a = a.replace(' ','')
#             data.append(a)
#
# print(data)
# with open("意图数据/知道.txt", "w") as f:
#     for i in data:
#         f.write(i)
#         f.write("\n")

# ben = 10000
# jia = 10000
# li = 0.3
#
# for i in range(20):
#     print("第{}年利润".format(i+1))
#     lirun = ben * (li)
#     print(lirun)
#     print("第{}年本".format(i + 2))
#     print(ben+lirun + jia)
#     ben = ben + lirun +jia

a = ["a", "b", "c"]
print(a.index("a"))
print(a.index("d"))