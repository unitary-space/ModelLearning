"""
本节我们使用内部的模型来训练
"""
import torch
import torch.nn as nn
import torch.optim as optim

# 1. 损失函数
def test01():
    # 初始化平方损失函数对象
    criterion = nn.MSELoss()
    y_pred = torch.randn(3, 5, requires_grad=True)
    y_true = torch.randn(3, 5)
    # 计算损失
    loss = criterion(y_pred, y_true)

    print(loss)
    # 该类实现了一个  __call__ 方法，因此对象可以类似函数来使用


# 2. 假设函数
def test02():
    # 创建线性模型对象，输入数据的特征必须有10个，输出的数据的特征要有5个
    model = nn.Linear(in_features=10, out_features=5)
    # 输入数据
    inputs = torch.randn(4, 10)
    # nn.Linear 以及实现了 __call__ 方法
    y_pred = model(inputs)
    print(y_pred.shape)


# 3. 优化方法
def test03():
    model = nn.Linear(in_features=10, out_features=5)
    optimizer = optim.SGD(model.parameters(), lr=1e-3)
    # 梯度清零
    optimizer.zero_grad()
    # 然后使用反向传播，更新参数

    # 更新模型参数
    optimizer.step()



if __name__ == '__main__':
    test02()
