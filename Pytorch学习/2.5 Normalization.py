import torch
import torch.nn as nn

# Dropout 丢弃一些随机的神经元

# 1. 创建和使用 Dropout
def test01():
    torch.manual_seed(0)
    # 初始化 DropOut 对象
    dropout = nn.Dropout(p=0.8)
    # 初始化数据
    inputs = torch.randint(0,10, size=[5,8], dtype=torch.float64)
    print(inputs)
    print('--------------------------------------------------------')
    # 将数据 Dropout，每个数据有 p 的概率变为 0，其它的元素进行缩放，缩放值为 1/(1-p)
    outputs = dropout(inputs)
    print(outputs)

# dropout 的丢弃对参数的影响
def test02():
    torch.manual_seed(0)
    # 初始化权重
    w = torch.randn((15, 1), requires_grad=True, dtype=torch.float32)
    # 初始化输入数据
    x = torch.randint(0, 10, size=[5, 15], dtype=torch.float32)

    # 计算目标函数
    y = x @ w
    y = y.sum()

    # 反向传播
    y.backward()
    print(w.grad.reshape(1,-1).squeeze().numpy())


# 不会更新没写参数，具有正则化的作用
def test03():
    torch.manual_seed(0)
    # 初始化权重
    w = torch.randn((15, 1), requires_grad=True,dtype=torch.float32)
    # 初始化输入数据
    x = torch.randint(0, 10, size=[5, 15], dtype=torch.float32)
    # 进行随机丢弃
    dropout = nn.Dropout(p=0.8)
    x = dropout(x)
    # 计算目标函数
    y = x @ w
    y = y.sum()

    # 反向传播
    y.backward()
    print(w.grad.reshape(1,-1).squeeze().numpy())




if __name__ == '__main__':
    test02()
    print("-" * 80)
    test03()
