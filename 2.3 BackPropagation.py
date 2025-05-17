import torch
import torch.nn as nn
import torch.optim as optim


# 1. 搭建网络，继承父类 nn.Module
class Net(nn.Module):
    def __init__(self):
        """
        此处的神经网络如下
             〇
         ↗       ↖
        〇       〇
        ↑   ↗↖   ↑
        〇       〇
        ↑   ↗ ↖  ↑
        〇       〇

        """
        # 手动调用父类的初始化参数
        super(Net, self).__init__()

        self.linear1 = nn.Linear(in_features=2, out_features=2)
        self.linear2 = nn.Linear(in_features=2, out_features=2)

        # 手动对网络参数进行初始化
        self.linear1.weight.data = torch.tensor([[0.15, 0.20], [0.25, 0.30]])
        self.linear2.weight.data = torch.tensor([[0.40, 0.45], [0.5, 0.55]])
        self.linear1.bias.data = torch.tensor([0.35, 0.35])
        self.linear2.bias.data = torch.tensor([0.6, 0.6])

    # 正向传播的函数，会定义正向传播的计算过程
    def forward(self, x):
        x = self.linear1(x)
        x = torch.sigmoid(x)

        x = self.linear2(x)
        x = torch.sigmoid(x)

        return x

if __name__ == '__main__':
    # 输入数据，二维列表。表示不同批次的输入
    inputs = torch.tensor([[0.15, 0.10]])
    # 真实值
    target = torch.tensor([[0.01, 0.99]])

    # 初始化对象
    net = Net()
    # 相当于直接调用了 forward 函数：父类实现了 __call__ 方法
    output = net(inputs)

    # 计算误差
    loss = torch.sum((output - target) ** 2) / 2

    # 构建优化器
    optimizer = optim.SGD(net.parameters(), lr=0.5)

    # 反向传播
    optimizer.zero_grad()
    loss.backward()

    # 参数更新
    optimizer.step()

    # 打印参数
    print(net.linear1.weight.grad.data)
    print(net.linear2.weight.grad.data)

    # 打印更新后的参数值
    print(net.state_dict())





