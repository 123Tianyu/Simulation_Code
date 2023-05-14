import torch
from tqdm import tqdm
from Data_Processing import Approximately_equal

def accuracy_regression(output, target, index, correct):
    # 计算模型准确率  output和target的尺寸都为 batch_size×64
    correct = torch.sum(Approximately_equal(output=output, target=target, index=index)).item() # 计算满足标签与输出之差的绝对值小于等于index的个数
    total = output.nelement()  # 获取张量中元素的个数
    return correct,total

def model_learning_regression(model, learning_data, model_loss, Accuracy, Loss, model_optimizer, epochs, index):
    # 回归模型的训练函数
    for epoch in tqdm(range(epochs)):
        train_loss, val_loss = [], []
        print(f'Local Training Round : {epoch + 1}')

        model.train() # 表明当前是在训练
        loss_train, total_train, correct_train = 0.0, 0.0, 0.0
        for batch_idx, (data, target) in enumerate(learning_data):
            model_output = model(data)
            model_Loss = model_loss(model_output, target)

            model_optimizer.zero_grad()
            model_Loss.backward()
            model_optimizer.step()

            loss = model_Loss.item()
            train_loss.append(loss)

            # 计算每一轮模型的训练准确率之和
            correct_train += \
            accuracy_regression(output=model_output, target=target, index=index, correct=correct_train)[0]
            total_train += \
            accuracy_regression(output=model_output, target=target, index=index, correct=correct_train)[1]

            train_loss_ave = sum(train_loss) / len(train_loss)
            train_acc_ave = correct_train / total_train

            if batch_idx % 10 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\t loss: {:.6f}  acc: {:.6f}%'.format(
                        epoch + 1, batch_idx * len(data), len(learning_data.dataset),
                        100. * batch_idx / len(learning_data),
                        train_loss_ave,  # 输出的前batch_idx个batch的平均损失值
                        100 * train_acc_ave))

    Accuracy.append(train_loss_ave)  # 获取本地第十个epoch的平均准确率
    Loss.append(train_loss_ave)      # 获取本地第十个epoch的平均损失值
    return model,Accuracy,Loss,train_loss_ave,train_acc_ave

def model_testing_regression(model, learning_data, model_loss, Accuracy, Loss, epochs, index):
    # 回归模型的测试函数
    for epoch in tqdm(range(epochs)):
        local_weights, local_loss = [], []
        print(f'Local Testing Round : {epoch + 1} \n')

        model.eval()  # 表明当前是在测试
        loss, total, correct = 0.0, 0.0, 0.0
        for batch_idx, (data, target) in enumerate(learning_data):
            model_output = torch.tensor(model(data), dtype=torch.float64)
            # print(target)
            # print(data.size())
            a=torch.zeros_like(model_output)
            # print(model_output)
            model_Loss = model_loss(model_output, target)

            loss = model_Loss.item()
            local_loss.append(loss)

            # 计算每一轮模型的训练准确率之和
            correct += accuracy_regression(output=model_output, target=target, index=index, correct=correct)[0]
            total   += accuracy_regression(output=model_output, target=target, index=index, correct=correct)[1]

            # correct += accuracy_regression(output=a, target=target, index=index, correct=correct)[0]
            # total   += accuracy_regression(output=a, target=target, index=index, correct=correct)[1]

            if batch_idx % 10 == 0:
                print('Test Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}  acc: {:.6f}%'.format(
                        epoch + 1, batch_idx * len(data), len(learning_data.dataset),
                        100. * batch_idx / len(learning_data),
                        sum(local_loss) / len(local_loss), # 输出的前batch_idx个batch的平均损失值 因此每个epoch开始就会归零
                        100 * correct / total))

        Accuracy.append(correct / total)
        Loss.append(sum(local_loss) / len(local_loss))
    return model,Accuracy,Loss

def predict(model, learning_data, model_loss, model_optimizer, epochs, index):
    # 回归模型的训练函数
    for epoch in tqdm(range(epochs)):
        train_loss, val_loss = [], []
        Model_output = []

        model.eval() # 表明当前是在训练
        loss_train, total_train, correct_train = 0.0, 0.0, 0.0
        for batch_idx, (data, target) in enumerate(learning_data):
            # print(type(data))
            model_output = model(data)
            Model_output.append(model_output)
            # print(model_output.shape)
            # print(target.shape)
            model_Loss = model_loss(model_output, target)

            model_optimizer.zero_grad()
            model_Loss.backward()
            model_optimizer.step()

            loss = model_Loss.item()
            train_loss.append(loss)

            # 计算每一轮模型的训练准确率之和
            correct_train += \
            accuracy_regression(output=model_output, target=target, index=index, correct=correct_train)[0]
            total_train += \
            accuracy_regression(output=model_output, target=target, index=index, correct=correct_train)[1]

        Model_acc = correct_train / total_train
        Model_loss = sum(train_loss) / len(train_loss)
    return Model_output,Model_loss,Model_acc



