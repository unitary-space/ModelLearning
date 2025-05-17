import torch
from sklearn.datasets import make_regression
import matplotlib.pyplot as plt
import random

plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号


# 构建数据集
def create_dataset():
    (x, y, coef) = make_regression(n_samples=100,
                                   n_features=1,
                                   noise=20,
                                   coef=True,
                                   bias=14.5,
                                   random_state=0)
    # 将构建的数据转化为张量
    x = torch.tensor(x)
    y = torch.tensor(y)

    return x, y, coef


# 构建数据加载器
def data_loader(x, y, batch_size):
    # 计算一下样本的数量
    data_len = len(y)

    # 构建数据索引
    data_index = list(range(data_len))

    # 数据集进行打乱
    random.shuffle(data_index)

    # 计算 batch 的数量
    batch_number = data_len // batch_size

    for idx in range(batch_number):
        start = idx * batch_size
        end = start + batch_size

        batch_train_x = x[start:end]
        batch_train_y = y[start:end]

        # 一个生成器
        yield batch_train_x, batch_train_y


def test01():
    train()


# 假设函数
w = torch.tensor(0, requires_grad=True, dtype=torch.float64)
b = torch.tensor(0, requires_grad=True, dtype=torch.float64)


def linear_regression(x):
    return w * x + b


# 损失函数
def square_loss(y_pred, y_true):
    return (y_pred - y_true) ** 2


# 优化方法，即更新参数，使用梯度下降法
def sgd(lr=1e-2):
    # 使用批次样本的平均值
    w.data = w.data - lr * w.grad.data / 16  # 16 是 batch_size
    b.data = b.data - lr * b.grad.data / 16


def train():
    # 家族数据集
    x, y, coef = create_dataset()
    # 定义训练参数
    epochs = 1000
    learning_rate = 1e-2
    epoch_loss = []
    total_loss = 0.0
    train_samples = 0

    for _ in range(epochs):
        for train_x, train_y in data_loader(x, y, batch_size=16):
            # 1.送入模型进行预测
            y_pred = linear_regression(train_x)
            # 2. 计算预测值和真实值之间的平方损失
            loss = square_loss(y_pred, train_y.reshape(-1, 1)).sum()  # 让 train_y 和 y_pred 形状一致
            total_loss += loss.item()
            train_samples += len(train_y)
            # 3. 梯度清零
            if w.grad is not None:
                w.grad.data.zero_()
            if b.grad is not None:
                b.grad.data.zero_()
            # 4. 自动微分
            loss.backward()

            # 5. 参数更新
            sgd(learning_rate)

            # 6. 输出损失
            print('loss: %10f' % (total_loss / train_samples))
        # 记录每一个 epoch 的平均损失
        epoch_loss.append(total_loss / train_samples)
    # 先绘制数据集散点图
    plt.scatter(x, y)
    # 绘制拟合直线
    x = torch.linspace(x.min(), x.max(), 1000)
    y1 = torch.tensor(([v * w + b for v in x]))
    y2 = torch.tensor(([v * coef + 14.5 for v in x]))

    plt.plot(x, y1, label='训练')
    plt.plot(x, y2, label='真实')
    plt.grid()
    plt.legend()
    plt.show()

    # 打印损失变化曲线
    plt.plot(range(epochs), epoch_loss)
    plt.grid()
    plt.title('损失变化曲线')
    plt.show()


if __name__ == '__main__':
    test01()
