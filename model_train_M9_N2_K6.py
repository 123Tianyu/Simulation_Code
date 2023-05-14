# 系统模型：一个边长为 D=1KM 的正方形区域内,分布着个数为 M=9 的AP, 天线数 N=2
# 以及随机均匀分布着个数为 K=6 的单天线用户的 Cell-free 大规模MIMO系统
import torch
import torch.nn as nn
from Training_func import model_learning_regression,model_testing_regression
from Data_Processing import data_get,Data_normalization,Global_integration_MRC
from Other_func import isexist,param_set,excel_write
from Model import DNN_Network
import torch.optim as optim
import numpy as np
import time

# 这块适用于程序运行时忽略掉警告信息，不让警告输出
import warnings
warnings.filterwarnings("ignore")

# # 获取matlab产生的数据
import hdf5storage
file = '.\Simulation_M9_N2_K6.mat'
data = hdf5storage.loadmat(file) # 导入后的data是一个字典，取出想要的变量字段即可

# 数据处理
Hhat_data_AP_vector = torch.tensor(data['Hhat_data_DNN_MMSE'],dtype=torch.float32)
V_MMSE_data_AP = torch.tensor(data['V_MMSE_global_DNN'],dtype=torch.float32)

# 设置参数
param = param_set(data)

# 获取训练数据以及测试数据
train_data  = Hhat_data_AP_vector[:,0:int(param["training_sets_index"]*Hhat_data_AP_vector.size(1)),:]
train_label = V_MMSE_data_AP[:,0:int(param["training_sets_index"]*V_MMSE_data_AP.size(1)),:]

test_data   = Hhat_data_AP_vector[:,int(param["training_sets_index"]*Hhat_data_AP_vector.size(1)):int(Hhat_data_AP_vector.size(1)),:]
test_label  = V_MMSE_data_AP[:,int(param["training_sets_index"]*V_MMSE_data_AP.size(1)):int(V_MMSE_data_AP.size(1)),:]

# 在指定路径中寻找之前训练保存的模型参数文件  存在则获取之前的模型参数继续训练  否则重新训练
Path = 'Model_param_MMSE_M9_N2_K6_simulation.pkl'
if not isexist(name=Path):
    exist_file = 0                # 当该变量为0 则说明当前路径下没有存储模型参数  因此全局模型权重为计算机算计分配
    global_model = DNN_Network(input_size=param['input_size'], output_size=param['output_size'])
    global_model_weights = {}
else:
    exist_file = 1                # 当该变量为1 则说明当前路径下有存储模型参数   因此加载之前的模型参数
    global_model_weights = torch.load(Path)
    global_model = DNN_Network(input_size=param['input_size'], output_size=param['output_size'])
    global_model.load_state_dict(global_model_weights)

# 设置网络训练参数
training_loss,training_acc = [[],[],[],[],[],[],[],[],[]],\
                             [[],[],[],[],[],[],[],[],[]]   # 存储各本地模型训练的平均损失值以及模型的平均训练精度（准确率）
training_loss_ave,training_acc_ave = [],[]  # 存储每轮参与联邦学习的用户平均损失值以及平均准确率
testing_loss,testing_acc = [],[]  # 存储各本地模型测试的平均损失值以及模型的平均训练精度（准确率）
local_model_loss = []
local_model_optimizer = []
local_model = []               # 存储各个本地设备的网络模型
local_model_weights = []       # 存储各个本地设备的网络模型权重
for u in range(param["User_num"]):
    local_model.append(global_model)
    local_model_weights.append(u)
    if exist_file==1:  # 当前路径文件下有存储模型参数的文件时，将之前训练存储的全局模型权重参数分配给各个本地用户模型
        local_model[u].load_state_dict(global_model_weights)
    local_model_loss.append(nn.MSELoss())  # nn.MSELoss()默认reduce = True，size_average = True
                                           # 即直接返回损失张量中所有元素的平均值 也就是平均损失值
    local_model_optimizer.append(optim.SGD(local_model[u].parameters(), lr=param["lr"])) # local_model[u].parameters()是传入网络参数的意思

# 将数据根据AP分配整合打包
Training_data = []
Testing_data  = []
for u in range(param["User_num"]):
    Training_data.append(data_get(Data_normalization(train_data[u])[0], Data_normalization(train_label[u])[0],param["batch_size"], shuffle_choice=True))
    Testing_data.append(data_get(Data_normalization(test_data[u])[0], Data_normalization(test_label[u])[0],param["batch_size"], shuffle_choice=True))

# 开始训练
time_begin = time.time()
for t in range(param["times"]):
    User_set = list(np.random.choice(param["User_num"], param["select_num"], replace=False)) # 每轮联邦学习随机选择一部分用户
    model_weights = []
    Train_loss = 0
    Train_acc = 0
    print(f"第{t + 1}轮联邦学习训练开始: ")
    # 分别对不同用户的本地模型进行训练
    for u in range(len(User_set)):
        print(f"本地用户{User_set[u]+1}开始训练: ")
        # 对本地用户u进行模型训练
        local_model[User_set[u]], training_acc[User_set[u]], training_loss[User_set[u]], train_loss_ave, train_acc_ave = \
            model_learning_regression(model=local_model[User_set[u]],
                                      learning_data=Training_data[User_set[u]],
                                      model_loss=local_model_loss[User_set[u]],
                                      model_optimizer=local_model_optimizer[User_set[u]],
                                      Accuracy=training_acc[User_set[u]],
                                      Loss=training_loss[User_set[u]],
                                      epochs=param["epochs_num"],
                                      index=param["acc_index"])
        Train_loss += train_loss_ave
        Train_acc += train_acc_ave
        local_model_weights[User_set[u]] = local_model[User_set[u]].state_dict()
        model_weights.append(local_model_weights[User_set[u]])
    training_loss_ave.append(Train_loss / len(User_set))
    training_acc_ave.append(Train_acc / len(User_set))
    # 将训练完的本地模型权重以及偏置进行全局聚合
    layer_param, local_model_layer_name = Global_integration_MRC(model_weights=model_weights, User_number=len(User_set))

    # 利用该循环将经过全局聚合后的权重与偏置信息整合到global_model_weights中
    for l in range(len(layer_param)):
        global_model_weights[local_model_layer_name[l]] = layer_param[l]

    # 更新本地模型权重与偏置
    for u in range(param["User_num"]):
        local_model[u].load_state_dict(global_model_weights)  # load_state_dict传入的量必须是字典，因此上面操作必不可少

    # 模型保存
    torch.save(global_model_weights, Path)

    print(f"第{t + 1}轮联邦学习训练结束: ")

# 模型测试
for u in range(param["User_num"]):
    print(f"本地用户{u+1}开始测试: ")
    _, testing_acc, testing_loss = \
        model_testing_regression(model=local_model[u],
                                 learning_data=Training_data[u],
                                 model_loss=local_model_loss[u],
                                 Accuracy=testing_acc,
                                 Loss=testing_loss,
                                 epochs=param["epochs_num"],
                                 index=param["acc_index"])

time_end = time.time()
all_time = time_end - time_begin
print(f"学习时间为: {all_time}")

# 训练数据存储
# 判断指定目录下是否存在该excel文件，不存在就重新创建并将数据写入excel文件；存在就先读取该文件中的内容到变量中  然后在变量中添加需要的值后再重新存入excel中
# 导入OS模块，用于判断是否已存在相应文件
import os
import xlrd


# 要复制的目标文件目录
cp_excel_file_path = ".\M9_N2_K6_Loss_Acc.xlsx"
# 如果已存在要创建的文件，删除（目的是可以让代码重复运行不出现已存在文件现象）
if os.path.exists(cp_excel_file_path):
    data1 = xlrd.open_workbook(cp_excel_file_path)
    sheet = data1.sheet_by_index(0)

    # 获取某一列的值
    loss_excel = sheet.col_values(0)[1:]
    loss_excel.extend(training_loss_ave)
    acc_excel = sheet.col_values(1)[1:]
    acc_excel.extend(training_acc_ave)
    datalist = [loss_excel,acc_excel]

    os.remove(cp_excel_file_path)
    print(f"{cp_excel_file_path}删除成功!")

    excel_write(datalist,['Loss','Acc'],cp_excel_file_path)

else:
    datalist = [training_loss_ave, training_acc_ave]
    excel_write(datalist,['Loss','Acc'],cp_excel_file_path)

