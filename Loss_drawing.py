import os
import xlrd

excel_loss = []
# 要复制的目标文件目录
cp_excel_file_path = ['M9_N1_K4_Loss_Acc.xlsx','M9_N1_K6_Loss_Acc.xlsx','M9_N2_K6_Loss_Acc.xlsx']
for i in range(len(cp_excel_file_path)):
    # 如果已存在要创建的文件，删除（目的是可以让代码重复运行不出现已存在文件现象）
    if os.path.exists(cp_excel_file_path[i]):
        data = xlrd.open_workbook(cp_excel_file_path[i])
        sheet = data.sheet_by_index(0)

        # 获取某一列的值
        excel_loss.append(sheet.col_values(0)[1:501])

    else:
        print(f"{cp_excel_file_path[i]}不存在, 请添加该文件后再运行!")


import matplotlib.pyplot as plt

font_dic = {"family":"Times New Roman",
            "weight":"normal",
            "size":12}    # 设置坐标轴标题的字体等参数

plt.figure(1)
line1, = plt.plot(range(1,len(excel_loss[0])+1), excel_loss[0], '-.', linewidth=2.5)
line2, = plt.plot(range(1,len(excel_loss[1])+1), excel_loss[1], ':', linewidth=2.5)
line3, = plt.plot(range(1,len(excel_loss[2])+1), excel_loss[2], '--', linewidth=2.5)
labels = ["M=1, K=4","M=1, K=6","M=2, K=6"]
handles = [line1, line2, line3]
from matplotlib.legend_handler import HandlerLine2D
handler = HandlerLine2D(numpoints=4)
plt.rcParams.update({'legend.fontsize': 11})
plt.legend(handles=handles, labels=labels, handler_map={line1: handler, line2: handler, line3: handler}, handlelength=3)

# plt.legend(["M=1, K=4","M=1, K=6","M=2, K=6"],loc='upper right')
plt.xlabel(r'$Number$ $of$ $iterations$', fontdict=font_dic)
plt.ylabel(r'$Value$ $of$ $Loss$ $Function$', fontdict=font_dic)
plt.grid(ls='--')
plt.show()

