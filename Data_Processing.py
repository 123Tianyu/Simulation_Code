import torch
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader

# 该函数用于将数据和标签打包在一块
def data_get(data, label, batchsize, shuffle_choice):
    data_ds = TensorDataset(data, label)
    data_dl = DataLoader(data_ds, batch_size=batchsize, shuffle=shuffle_choice, num_workers=0)
    return data_dl


def Data_normalization(Data):
    data = Data.clone()
    Max = torch.zeros(data.size(0))
    Min = torch.zeros(data.size(0))
    for i in range(data.size(0)):
        Max[i] = torch.max(data[i])
        Min[i] = torch.min(data[i])
        for j in range(data.size(1)):
            data[i][j] = (data[i][j] - Min[i]) / (Max[i] - Min[i])
    return data, Max, Min

def Data_denormalizing(data, Max, Min):
    # 该函数用于对四维张量去归一化，用于将输出值去归一化
    Data = data.clone()
    for i in range(data.size(0)):
        for j in range(data.size(1)):
            Data[i][j] = Data[i][j]*(Max[i] - Min[i]) + Min[i]
    return Data

def Approximately_equal(output, target, index):
    # 该函数用于判断张量在满足最大误差条件下的近似相等
    Compare_tensors = torch.zeros_like(output)  # 创建一个和output形状一样的全零tensor
    for i in range(output.size(0)):
        for j in range(output.size(1)):
                if (torch.abs(output[i][j] - target[i][j]) < index):  # 当相同位置的标签值与输出值之差的绝对值小于index则为真，否则为假
                    Compare_tensors[i][j] = True
                else:
                    Compare_tensors[i][j] = False
    return Compare_tensors

def Global_integration_MRC(model_weights, User_number):
    # 该函数用于将本地训练后的权重和偏置进行聚合
    layer_param = [] # 用于存储各层的权重与偏置
    length = len(model_weights[0]) # 计算 layer_param 长度

    model_weights_key = [] # 存储
    for key in model_weights[0].keys():
        model_weights_key.append(key)
    for l in range(length):
        # size.append(model_weights[0][l].shape)
        layer_param.append(torch.zeros(model_weights[0][model_weights_key[l]].shape))

    for k in range(User_number):
        for l in range(length):
            layer_param[l] += model_weights[k][model_weights_key[l]] / User_number

    return layer_param,model_weights_key

def Global_integration_MMSE(model_weights, User_number, User_set, Aggregation_coefficient):
    # 该函数用于将本地训练后的权重和偏置进行聚合
    layer_param = [] # 用于存储各层的权重与偏置
    length = len(model_weights[0]) # 计算 layer_param 长度

    model_weights_key = [] # 存储
    for key in model_weights[0].keys():
        model_weights_key.append(key)
    for l in range(length):
        # size.append(model_weights[0][l].shape)
        layer_param.append(torch.zeros(model_weights[0][model_weights_key[l]].shape))

    for k in range(User_number):
        for l in range(length):
            layer_param[l] += Aggregation_coefficient[User_set[k]]*model_weights[k][model_weights_key[l]]

    return layer_param,model_weights_key