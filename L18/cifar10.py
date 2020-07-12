import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt 
import torchvision.transforms as transforms
import numpy as np
import time

EPOCH = 10
BATCH_SIZE = 16

train_data = datasets.CIFAR10(root='C:/Users/zero/Desktop/课前资料/martech课前资料/cifar10/',
                            train=True,
                            transform=transforms.ToTensor(),
                            download=False)
test_data = datasets.CIFAR10(root='C:/Users/zero/Desktop/课前资料/martech课前资料/cifar10/',
                            train=False,
                            transform=transforms.ToTensor(),
                            download=False)
print(train_data.data.shape)
# temp = train_data[1][0].numpy()
# print(temp.shape)
# temp = temp.transpose(1,2,0)
# print(temp.shape)
# plt.imshow(temp)
# plt.show()

# 使用DataLoader进行分批
train_loader = DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(dataset=test_data, batch_size=BATCH_SIZE, shuffle=True)

model = torchvision.models.densenet161(pretrained=True)
# #损失函数:这里用交叉熵
# criterion = nn.CrossEntropyLoss()
# #优化器 这里用SGD
# optimizer = optim.Adam(model.parameters(), lr=1e-3)

# #device : GPU or CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# model.to(device)

# # 训练
# for epoch in range(EPOCH):
#     start_time = time.time()
#     for i, data in enumerate(train_loader):
#         inputs, labels = data
#         inputs, labels = inputs.to(device), labels.to(device)
#         # 前向传播
#         outputs = model(inputs)
#         # 计算损失函数
#         loss = criterion(outputs, labels)
#         # 清空上一轮梯度
#         optimizer.zero_grad()
#         # 反向传播
#         loss.backward()
#         # 参数更新
#         optimizer.step()
#     print('epoch{} loss:{:.4f} time:{:.4f}'.format(epoch+1, loss.item(), time.time()-start_time))

# #保存训练模型
file_name = 'cifar10_resnet.pt'
# torch.save(model, file_name)
# print(file_name+' saved')


# 测试
model = torch.load(file_name)
model.eval()
correct, total = 0, 0

for data in test_loader:
    images, labels = data
    images, labels = images.to(device), labels.to(device)
    # 前向传播
    out = model(images)
    _, predicted = torch.max(out.data, 1)
    total += labels.size(0)
    correct += (predicted == labels).sum().item()

#输出识别准确率
print('10000测试图像 准确率:{:.4f}%'.format(100.0 * correct / total)) 