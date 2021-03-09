import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class Net(nn.Module):
    def __init__(self):               #输入1x32x32
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1,6,3) #输入频道1，输出6，卷积3x3
        self.conv2 = nn.Conv2d(6,16,3)
        #全连接
        self.fc1 = nn.Linear(16*28*28,512)  # 输入维度：16*28*28 其中28=32经过两次卷积得到的
        self.fc2 = nn.Linear(512,64)
        self.fc3 = nn.Linear(64,2)

    def forward(self,x):  #定义数据流向
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)

        x = x.view(-1,16*28*28) #将形状变为此维度
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)

        return x

def get_data():  #获取数据
    input_data = torch.randn(1,1,32,32)
    print(input_data)
    print("input_data.size()："+str(input_data.size()))
    print("input_data.shape："+str(input_data.shape))

    target = torch.randn(2)  #随机生成真实值
    target = target.view(1,-1)

    return input_data,target

def data_loss(out, target):
    criterion = nn.L1Loss()       #定义损失函数
    loss = criterion(out,target)  #计算损失
    print(loss)
    return loss
    pass


#训练
# def fit_model():
#     criterion = nn.CrossEntropyLoss()      # 定义损失函数,交叉熵
#     optimizer = optim.SGD(net.parameters, lr=0.001, momentum=0.9)  # 权重更新规则
#     for epoch in range(2):
#         for i, data in enumerate(trainloader):
#             images, labels = data
#             output = net(images)
#
#             loss = criterion(outputs, labels)  #计算损失




if __name__ == "__main__":
    net = Net()
    print(net)
    input_data,target = get_data()
    out = net(input_data)  #运行神经网络
    print(out)
    print(out.size())
    loss = data_loss(out,target)

    #反向传递
    net.zero_grad()  #梯度清零
    loss.backward()  #自动计算梯度，反向清零
    optimizer = optim.SGD(net.parameters, lr=0.001)
    optimizer.step()




