import torch


if __name__ == '__main__':
    x = torch.tensor(1.0)
    w = torch.tensor(0.0, requires_grad=True)
    b = torch.tensor(0.0, requires_grad=True)

    # 构建计算过程
    f = (1 + torch.exp(-(w*x + b))) ** -1

    # 计算导数
    f.backward()

    # 打印
    print(w.grad)


    # 学习率
    lr = 1e-2

    # ......