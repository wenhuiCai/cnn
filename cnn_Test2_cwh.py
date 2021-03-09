#CIFAR10数据集 10个类6万张图
import torch
import torchvision
import torchvision.transforms as transforms  #归一化库
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

def create_data():
    # 创建
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # 归一标准化:RGB颜色通道的平均值，RGB颜色的标准偏差
            # transforms.flit()   #翻转
        ]
    )
    #私人数据集
    # images_path = 'E:/data/train'
    # privateset = torchvision.datasets.ImageFolder(root=images_path, transform=transform)
    # pri_dataloader = torch.utils.data.DataLoader(privateset,batch_size=4, shuffle=True, num_worker=2)

    # 训练数据集 #加载方法
    trainset = torchvision.datasets.CIFAR10(root="E:/data", train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True, num_workers=2)
    # 测试数据集 #加载方法
    testset = torchvision.datasets.CIFAR10(root="E:/data", train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=4, shuffle=False, num_workers=2)
    return trainloader, testloader
#显示数据
def show_data(img):
    img = img/2 + 0.5  #反归一
    nping = img.numpy()
    nping = np.transpose(nping,(1,2,0))
    plt.imshow(nping)

class Net(nn.Module):
    def __init__(self):               #输入3x32x32
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3,6,3) #输入频道3，输出6，卷积3x3
        self.conv2 = nn.Conv2d(6,16,3)
        #全连接
        self.fc1 = nn.Linear(16*28*28,512)  # 输入维度：16*28*28 其中28=32经过两次卷积得到的
        self.fc2 = nn.Linear(512,64)
        self.fc3 = nn.Linear(64,10)  #输入输出维度

    def forward(self,x):  #定义数据流向
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)

        x = x.view(-1,16*28*28)   #将形状变为此维度
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        return x

#更新权重训练
def update_regulation(net,trainloader):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    for epoch in range(2):
        for i, data in enumerate(trainloader):
            images, labels = data
            outputs =net(images)
            loss = criterion(outputs, labels)    #计算损失
            optimizer.zero_grad()                #更新网络权重的第一步，将所有梯度清零
            loss.backward()                      #将学习的梯度传到所有的网络权重上
            optimizer.step()                     #根据梯度更新所有权值

            if(i%1000 ==0):
                print('Epoch: %d, Step: %d, Loss: %.3f' %(epoch,i,loss.item()))
#测试模型
def test_model(testloader):
    correct = 0.0
    total = 0.0

    with torch.no_grad():
        for data in testloader:
            images, labels = data
            outputs = net(images)  #输出的是10个类的概率
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted ==labels).sum()
            total += labels.size(0)

    print('准确率：',float(correct)/total)


if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    create_data()
    #显示数据
    trainloader, testloader = create_data().to(device)
    dataiter = iter(trainloader)
    images, labels = dataiter.next().to(device)
    show_data(torchvision.utils.make_grid(images))
    net = Net().to(device)
    update_regulation(net,trainloader)  #权重更新规则
    test_model(testloader)
    print(net)
    #保存模型
    torch.save(net.state_dict(), 'E:/data/model.pt')
    #读取模型
    #net.load_state_dict(torch.load('E:/data/model.pt'))



