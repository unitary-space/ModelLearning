import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

"""
创建神经网络类：
1. 创建 Net 类，继承 nn.Module
2. 在 Net 类的 __init__(self) 方法中，使用线性模型定义神经网络的基本信息：每一层的输入和输出的特征数
3. 在 Net 类中，定义前向传播函数 forward(self, x) ，其中 x 是输入的向量。里面是非线性的函数，即这个网络运行一次的计算过程
"""


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(28*28, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 64)
        self.fc4 = nn.Linear(64, 10)

    def forward(self, x):

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.log_softmax(self.fc4(x), dim=1)

        return x


# 获得 MNIST 数据集
def get_data_loader(is_train):
    to_tensor = transforms.Compose([transforms.ToTensor()])
    data_set = MNIST("", is_train, transform=to_tensor, download=True)
    return DataLoader(data_set, batch_size=15, shuffle=True)


# 评估函数
def evaluate(test_data, net):
    n_correct = 0
    n_total = 0
    with torch.no_grad():
        for (x, y) in test_data:
            outputs = net.forward(x.view(-1, 28*28))
            for i, output in enumerate(outputs):
                if torch.argmax(output) == y[i]:
                    n_correct += 1
                n_total += 1
    return n_correct / n_total

def main():
    print('CUDA 可以运行' if torch.cuda.is_available() else '只能在 CPU 上训练')
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_data = get_data_loader(is_train=True)
    test_data = get_data_loader(is_train=False)
    net = Net()


    print("Initial accuracy:", evaluate(test_data, net))
    optimizer = optim.Adam(net.parameters(), lr=1e-3)
    for epoch in range(10):
        for (x, y) in train_data:
            net.zero_grad()
            output = net.forward(x.view(-1, 28*28))
            loss = F.nll_loss(output, y) # 对数损失函数
            loss.backward()
            optimizer.step()
        print("epoch", epoch, "accuracy:", evaluate(test_data, net))

    for (n, (x, _)) in enumerate(test_data):
        if n > 2:
            break

        predict = torch.argmax(net.forward(x[0].view(-1, 28*28)))
        plt.figure(n)
        plt.imshow(x[0].view(28, 28))
        plt.title('prediction ' + str(int(predict)))
    plt.show()

if __name__ == '__main__':
    main()