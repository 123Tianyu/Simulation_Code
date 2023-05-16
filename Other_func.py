import os
import numpy as np
import xlwt

def param_set(data):
    # 设置参数
    param = {
        "M": int(data['M']),                        # AP数
        "N": int(data['N']),                        # AP天线数
        "K": int(data['K']),                        # 单天线用户数
        "Sample_num" : int(data["Sample_num"]),     # 样本数
        "training_sets_index": 0.5,                 # 训练集占全部数据集的比例
        "times": 2000,                               # 进行联邦学习的次数
        "epochs_num": 10,                           # 迭代轮数
        "batch_size": 64,                           # mini-batch的大小
        "lr": 1e-1,                                 # 学习率
        "acc_index": 1e-2,                          # 计算准确率判断为正确所允许的最大误差值
        "User_num": int(data['M']),                 # 联邦学习参与训练的AP集长度
        "select_num": 5,                            # 每轮联邦学习参与的AP个数
        "predict_data_len": int(0.5*data["Sample_num"]),  # 用于预测的数据长
        "input_size": int(3 * data['K'] * data['N']),  # 输入层神经元个数
        "output_size": int(2 * data['K'] * data['N']),  # 输出层神经元个数
        "Product_factor": (data['tau_c'] - data['tau_p']) / data['tau_c']
    }
    return param

def excel_write(datalist,col,savepath):  # 创建一个新的excel表并写入数据
    book = xlwt.Workbook(encoding='utf-8', style_compression=0)  # 创建excel表格类型文件
    sheet = book.add_sheet('每轮联邦学习得到的平均损失值和准确率', cell_overwrite_ok=True)  # 在excel表格类型文件中建立一张sheet表单
    for i in range(len(col)):  # 将列属性元组col写进sheet表单中
        sheet.write(0, i, col[i])

    for i in range(0, len(col)):  # 将数据写进sheet表单中
        Data = datalist[i]
        for j in range(len(Data)):
            sheet.write(j + 1, i, Data[j])

    # 保存excel文件
    book.save(savepath)
    print('目标文件不存在，已成功生成一个文件')

def isexist(name, path=None): # 该函数用于判断当前路径中文件是否存在
    '''
    :param name: 需要检测的文件或文件夹名
    :param path: 需要检测的文件或文件夹所在的路径，当path=None时默认使用当前路径检测
        :return: True/False 当检测的文件或文件夹所在的路径下有目标文件或文件夹时返回Ture,
                当检测的文件或文件夹所在的路径下没有有目标文件或文件夹时返回False
    '''
    if path is None:
        path = os.getcwd()
        if os.path.exists(path + '/' + name):
            print("Under the path: " + path + '\n' + name + " is exist")
            return True
        else:
            if (os.path.exists(path)):
                print("Under the path: " + path + '\n' + name + " is not exist")
            else:
                print("This path could not be found: " + path + '\n')
            return False

def Compute_SE_uplink_M(Hhat_mk,V_mk,K,M,yinta,pu_cf):  # 计算多天线速率
    Hhat_mk = np.squeeze(Hhat_mk) # (M,N,K)
    V_mk    = np.squeeze(V_mk)    # (M,N,K)
    Ru_cf_k = np.zeros((1, K))
    # 计算速率
    for k in range(K):
        temp1 = 0
        for m in range(M):
            temp1 = temp1 + V_mk[m,:,k].T.conjugate() * Hhat_mk[m,:,k]
        Ru_cf_k_fenzi = pu_cf * yinta[0,k] * np.linalg.norm(temp1) ** 2

        Ru_cf_k_fenmu_left = 0
        temp2 = 0
        for m in range(M):
            temp2 = temp2 + V_mk[m, :, k].T.conjugate()
        Ru_cf_k_fenmu_right = np.linalg.norm(temp2)**2

        for k1 in range(K):
            if (k1!=k):
                temp3 = 0
                for m in range(M):
                    temp3 = temp3 + V_mk[m, :, k].T.conjugate() * Hhat_mk[m, :, k1]

                Ru_cf_k_fenmu_left = Ru_cf_k_fenmu_left + yinta[0,k1] * np.linalg.norm(temp3)**2

        Ru_cf_k_fenmu_left = pu_cf * (Ru_cf_k_fenmu_left)
        Ru_cf_k_fenmu_left = Ru_cf_k_fenmu_left[0][0]

        Ru_cf_k_fenmu = Ru_cf_k_fenmu_left + Ru_cf_k_fenmu_right
        Ru_cf_k[0,k] = np.log2(1 + Ru_cf_k_fenzi / Ru_cf_k_fenmu)
    return Ru_cf_k

def Compute_SE_uplink_S(Hhat_mk, V_mk, K, M, yinta, pu_cf):  # 计算单天线速率
    Hhat_mk = np.squeeze(Hhat_mk)  # (M,K)
    V_mk    = np.squeeze(V_mk)  # (M,K)
    Ru_cf_k = np.zeros((1, K))
    # 计算速率
    for k in range(K):
        temp1 = 0
        for m in range(M):
            temp1 = temp1 + V_mk[m,k].T.conjugate() * Hhat_mk[m,k]
        Ru_cf_k_fenzi = pu_cf * yinta[0, k] * np.linalg.norm(temp1) ** 2

        Ru_cf_k_fenmu_left = 0
        temp2 = 0
        for m in range(M):
            temp2 = temp2 + V_mk[m,k].T.conjugate()
        Ru_cf_k_fenmu_right = np.linalg.norm(temp2) ** 2

        for k1 in range(K):
            if (k1 != k):
                temp3 = 0
                for m in range(M):
                    temp3 = temp3 + V_mk[m,k].T.conjugate() * Hhat_mk[m,k1]

                Ru_cf_k_fenmu_left = Ru_cf_k_fenmu_left + yinta[0, k1] * np.linalg.norm(temp3) ** 2

        Ru_cf_k_fenmu_left = pu_cf * (Ru_cf_k_fenmu_left)
        Ru_cf_k_fenmu_left = Ru_cf_k_fenmu_left[0][0]

        Ru_cf_k_fenmu = Ru_cf_k_fenmu_left + Ru_cf_k_fenmu_right
        Ru_cf_k[0, k] = np.log2(1 + Ru_cf_k_fenzi / Ru_cf_k_fenmu)
    return Ru_cf_k

