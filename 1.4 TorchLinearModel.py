import torch
from sklearn.datasets import make_regression
from torch.utils.data import TensorDataset
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号


def create_dataset():
    (x, y, coef) = make_regression(n_samples=100,
                                   n_features=1,
                                   noise=10,
                                   coef=True,
                                   bias=14.5,
                                   random_state=0)
    # 将构建的数据转化为张量
    x = torch.tensor(x, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32)

    return x, y, coef


def train():
    # 1. 构建数据集
    x, y, coef = create_dataset()
    # 2. 构建数据集对象
    dataset = TensorDataset(x, y)
    # 3. 构建数据加载器
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
    # 4. 构建模型
    model = nn.Linear(in_features=1, out_features=1)
    # 5. 损失函数
    criterion = nn.MSELoss()
    # 6.优化方法
    optimizer = optim.SGD(model.parameters(), lr=1e-2)
    # 7. 初始化训练参数
    epochs = 100

    for _ in range(epochs):
        for train_x, train_y in dataloader:
            # 将一个 batch 的数据送入模型，要求是 float32 类型
            y_pred = model(train_x)
            # 计算损失
            loss = criterion(y_pred, train_y.reshape(-1, 1))
            # 梯度清零
            optimizer.zero_grad()
            # 自动微分
            loss.backward()
            # 更新参数
            optimizer.step()
    # 绘制拟合直线
    plt.scatter(x, y)
    x = torch.linspace(x.min(), x.max(), 1000)
    y1 = torch.tensor([v * model.weight + model.bias for v in x])
    y2 = torch.tensor([v * coef + 14.5 for v in x])

    plt.plot(x, y1, label='训练')
    plt.plot(x, y2, label='真实')
    plt.grid()
    plt.legend()
    plt.show()


if __name__ == '__main__':
    train()
