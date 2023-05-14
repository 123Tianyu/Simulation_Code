import torch
import torch.nn as nn
from Training_func import predict
from Data_Processing import data_get,Data_normalization,Data_denormalizing
from Other_func import isexist,Compute_SE_uplink_M,Compute_SE_uplink_S,param_set
from Model import DNN_Network
import torch.optim as optim
import numpy as np

# 这块适用于程序运行时忽略掉警告信息，不让警告输出
import warnings
warnings.filterwarnings("ignore")


import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

# # 获取matlab产生的数据
import hdf5storage
file = '.\Simulation_M9_N1_K4.mat'
data = hdf5storage.loadmat(file)
# 导入后的data是一个字典，取出想要的变量字段即可。

# 数据处理
Hhat_data_AP_vector = torch.tensor(data['Hhat_data_DNN_MMSE'],dtype=torch.float32)
V_MMSE_data_AP = torch.tensor(data['V_MMSE_global_DNN'],dtype=torch.float32)

# 设置参数
param = param_set(data)

# 获取训练数据以及测试数据
train_data  = Hhat_data_AP_vector[:,0:int(param["training_sets_index"]*param["Sample_num"]),:]
train_label = V_MMSE_data_AP[:,0:int(param["training_sets_index"]*param["Sample_num"]),:]

test_data   = Hhat_data_AP_vector[:,int(param["training_sets_index"]*param["Sample_num"]):int(param["Sample_num"]),:]
test_label  = V_MMSE_data_AP[:,int(param["training_sets_index"]*param["Sample_num"]):int(param["Sample_num"]),:]


# 设置网络训练参数
local_model_loss_MRC = []
local_model_optimizer_MRC = []

# 在指定路径中寻找之前训练保存的模型参数文件  存在则获取之前的模型参数继续训练  否则重新训练
Path = 'Model_param_MMSE_M9_N1_K4_simulation.pkl'
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
training_loss,training_acc = [],[]  # 存储各本地模型训练的平均损失值以及模型的平均训练精度（准确率）
testing_loss,testing_acc = [],[]
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

Predict_data = []
predict_data_Max, predict_data_Min = [],[]
predict_label_Max, predict_label_Min = [],[]

for u in range(param["User_num"]):
    data_normalization, data_Max, data_Min = Data_normalization(Hhat_data_AP_vector[u])
    label_normalization, label_Max, label_Min = Data_normalization(V_MMSE_data_AP[u])

    Predict_data.append(data_get(data_normalization[0:param["predict_data_len"],:],
                         label_normalization[0:param["predict_data_len"],:],param["batch_size"], shuffle_choice=False))
    predict_data_Max.append(data_Max)
    predict_data_Min.append(data_Min)
    predict_label_Max.append(label_Max)
    predict_label_Min.append(label_Min)

# 模型预测  将全部数据作为预测数据因为这样才能与数据集长度对应
print("模型预测: ")
v_kl_predict = [] # 存储联邦学习所得的所有AP的MMSE预测输出
predict_loss = []
predict_acc = []

# 分别对不同用户的本地模型进行训练
for u in range(param["User_num"]):
    print(f"本地用户{u+1}开始进行预测: ")
    # 对本地用户u进行模型训练
    v_kl_temp, predict_loss_temp, predict_acc_temp = predict(model=local_model[u],
                                              learning_data=Predict_data[u],
                                              model_loss=local_model_loss[u],
                                              model_optimizer=local_model_optimizer[u],
                                              epochs=param["epochs_num"], index=param["acc_index"])
    v_kl_predict.append(v_kl_temp)
    predict_loss.append(predict_loss_temp)
    predict_acc.append(predict_acc_temp)

# 将预测输出依据节点划分拼接
V_kl_predict_cat = []
for u in range(param["User_num"]):
    temp1 = torch.cat((v_kl_predict[u][0],v_kl_predict[u][1]),dim=0)
    for l in range(2,len(v_kl_predict[0])):
        temp1 = torch.cat((temp1, v_kl_predict[u][l]), dim=0)
    V_kl_predict_cat.append(temp1)


# 数据处理
# 数据去归一化
print("数据去归一化: ")
V_kl_MMSE = []
for u in range(param["User_num"]):
    V_kl_MMSE.append(Data_denormalizing(data=V_kl_predict_cat[u], Max=predict_label_Max[u], Min=predict_label_Min[u]))

# 数据还原
# V_mk = np.zeros((param["M"],param["predict_data_len"],param["K"]*param["N"]),dtype=complex)
V_mk = np.zeros((param["M"],param["predict_data_len"],param["K"]*param["N"]),dtype=complex)
for l in range(param["M"]):
    V_mk[l, :, :] = V_kl_MMSE[l][:, 0:param["K"]*param["N"]].detach().numpy() + \
                                1j * V_kl_MMSE[l][:,param["K"]*param["N"]:2*param["K"]*param["N"]].detach().numpy()


# temp1 = np.zeros((param["M"],param["predict_data_len"],param["K"],param["N"]),dtype=complex)
temp1 = np.zeros((param["M"],param["predict_data_len"],param["K"],param["N"]),dtype=complex)
for k in range(param["K"]):
    temp1[:, :, k, :] = V_mk[:, :, k*param["N"]:(k + 1)*param["N"]]
V_MMSE_predict = temp1.transpose((0,3,2,1))  # (M,N,K,Sample_num)


# 参数准备
pu_cf = data['pu_cf']
yinta = data['yinta']


Hhat_local = data['H_mk'] # (M, N, K, Sample_num)
V_MMSE_Centralized = data['V_MMSE_Global']
V_MMSE_Local = data['V_MMSE_Local'].transpose((0,2,1))
V_MRC_Local = data['V_MRC_Local'].transpose((0,2,1))

# rate_k_centralized_mmse_predict = np.zeros((param["predict_data_len"],param["K"])) # 集中式MMSE预测速率
rate_k_centralized_mmse_predict = np.zeros((param["predict_data_len"],param["K"])) # 集中式MMSE预测速率
rate_k_centralized_mmse = np.zeros((param["Sample_num"],param["K"]))               # 集中式MMSE速率
rate_k_local_mmse = np.zeros((param["Sample_num"],param["K"]))                     # 本地MMSE速率
rate_k_local_mrc = np.zeros((param["Sample_num"],param["K"]))                      # 本地MRC速率

for t in range(param["predict_data_len"]):
    if (param["N"]==1):
        rate_k_centralized_mmse_predict[t, :] = Compute_SE_uplink_S(Hhat_local[:, :, :, t], V_MMSE_predict[:, :, :, t], param["K"], param["M"],yinta, pu_cf)
    else:
        rate_k_centralized_mmse_predict[t, :] = Compute_SE_uplink_M(Hhat_local[:, :, :, t], V_MMSE_predict[:, :, :, t], param["K"], param["M"],yinta, pu_cf)

for t in range(param["Sample_num"]):
    if (param["N"]==1):
        rate_k_centralized_mmse[t,:] = Compute_SE_uplink_S(Hhat_local[:, :, :, t], V_MMSE_Centralized[:, :, :, t], param["K"], param["M"],yinta, pu_cf)
        rate_k_local_mmse[t, :] = Compute_SE_uplink_S(Hhat_local[:, :, :, t], V_MMSE_Local[:, :, t], param["K"], param["M"], yinta, pu_cf)
        rate_k_local_mrc[t, :] = Compute_SE_uplink_S(Hhat_local[:, :, :, t], V_MRC_Local[:, :, t], param["K"], param["M"], yinta, pu_cf)
    else:
        rate_k_centralized_mmse[t,:] = Compute_SE_uplink_M(Hhat_local[:, :, :, t], V_MMSE_Centralized[:, :, :, t], param["K"], param["M"],yinta, pu_cf)
        rate_k_local_mmse[t, :] = Compute_SE_uplink_M(Hhat_local[:, :, :, t], V_MMSE_Local[:, :, t],param["K"], param["M"], yinta, pu_cf)
        rate_k_local_mrc[t, :] = Compute_SE_uplink_M(Hhat_local[:, :, :, t], V_MRC_Local[:, :, t], param["K"], param["M"], yinta, pu_cf)


# 求和速率
rate_centralized_mmse_predict = np.sum(rate_k_centralized_mmse_predict,1)   # 联邦学习计算的和速率
rate_centralized_mmse = np.sum(rate_k_centralized_mmse,1)                   # 集中式MMSE和速率
rate_local_mmse = np.sum(rate_k_local_mmse,1)                               # 本地MMSE和速率
rate_local_mrc = np.sum(rate_k_local_mrc,1)                                 # 本地MRC和速率

# 将速率变量转为列表
rate_centralized_mmse_predict = rate_centralized_mmse_predict.tolist()
rate_centralized_mmse = rate_centralized_mmse.tolist()
rate_local_mmse = rate_local_mmse.tolist()
rate_local_mrc = rate_local_mrc.tolist()


# 保存数据到excel
# 要复制的目标文件目录
from Other_func import excel_write
import os
cp_excel_file_path = ".\M9_N1_K4_rate.xlsx"

# 如果已存在要创建的文件，删除（目的是可以让代码重复运行不出现已存在文件现象）
datalist = [rate_centralized_mmse,rate_local_mmse,rate_local_mrc,rate_centralized_mmse_predict]
if os.path.exists(cp_excel_file_path):
    os.remove(cp_excel_file_path)
    excel_write(datalist,['rate_centralized_mmse','rate_local_mmse','rate_local_mrc','rate_centralized_mmse_predict'],cp_excel_file_path)
else:
    excel_write(datalist, ['rate_centralized_mmse','rate_local_mmse','rate_local_mrc','rate_centralized_mmse_predict'], cp_excel_file_path)

# 画图
text_str = r"$M = 1$, $K = 4$"
x_site = [2.6307,2.7568,3.0218,3.3013]
y_site = [0.05,0.05,0.05,0.05]
yticks_step = [0,0.05,0.2,0.4,0.6,0.8,1]
labels = [0,0.05,0.2,0.4,0.6,0.8,1]
font_dic = {"family":"Times New Roman",
            "weight":"normal",
            "size":12}    # 设置坐标轴标题的字体等参数
import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerLine2D
line1, = plt.plot(np.sort(np.reshape(rate_centralized_mmse_predict,[param["predict_data_len"],1]),axis=0),np.linspace(0,1,param["predict_data_len"]),'g--',linewidth=2.5)
line2, = plt.plot(np.sort(np.reshape(rate_centralized_mmse,[param["Sample_num"],1]),axis=0),np.linspace(0,1,param["Sample_num"]),'y-',linewidth=2.5)
line3, = plt.plot(np.sort(np.reshape(rate_local_mmse,[param["Sample_num"],1]),axis=0),np.linspace(0,1,param["Sample_num"]),'c-.',linewidth=2.5)
line4, = plt.plot(np.sort(np.reshape(rate_local_mrc,[param["Sample_num"],1]),axis=0),np.linspace(0,1,param["Sample_num"]),'r:',linewidth=2.5)
plt.xlabel(r'$Spectral$ $Efficiency$ $[bit/s/Hz]$',fontdict=font_dic)
plt.ylabel(r'$Cumulative$ $distribution$ $function$',fontdict=font_dic)
plt.xlim(0,max(rate_centralized_mmse)+1)

legend_labels = ["FL-Aided MMSE", "Centralized MMSE", "Local MMSE", "Local MRC"]
handles = [line1, line2, line3, line4]
handler = HandlerLine2D(numpoints=4)
plt.rcParams.update({'legend.fontsize': 11})
plt.legend(handles=handles, labels=legend_labels,
           handler_map={line1: handler, line2: handler, line3: handler, line4: handler},
           handlelength=2.8, loc='upper left',frameon=True)
plt.text(10, 0.4, text_str, ha='center', va='center', fontsize=15, color='black')
plt.yticks(yticks_step,labels,rotation = 0)
plt.scatter(x_site, y_site, color='black', s=15, zorder=3)  # 在最大值点上绘制一个红色的圆点
plt.grid(ls='--')
plt.show()

