"""
我们来学习更好的优化方法，以便处理鞍点、局部极小值的问题
我们先来介绍指数加权移动平均（EMWA），它是一种平均值，使得具体当前数值距离越远的贡献越小，
具体越近的贡献越大
"""
import torch
import matplotlib.pyplot as plt

ELEMENT_NUMBER = 30


def test01():

    # 固定随机数种子
    torch.manual_seed(0)

    # 随机产生 31 天的温度
    temperature = torch.randn(size=[31]) * 10

    # 绘制温度的曲线
    days = torch.arange(1, 32, 1)
    plt.plot(days, temperature, 'o-r')
    plt.show()

def test02(beta=0.9):
    # 固定随机数种子
    torch.manual_seed(0)

    # 随机产生 31 天的温度
    temperature = torch.randn(size=[31]) * 10

    # 绘制温度的曲线
    days = torch.arange(1, 32, 1)

    # 存储历史的指数加权平均值
    exp_weight_avg = []

    for idx, temp in enumerate(temperature, 1):

        if idx == 1:
            exp_weight_avg.append(temp)
            continue

        new_temp = exp_weight_avg[idx - 2] * beta + (1 - beta) * temp
        exp_weight_avg.append(new_temp)

    # 绘制指数加权平均温度
    days = torch.arange(1, 32, 1)
    plt.plot(days, exp_weight_avg, 'o-r')
    plt.show()



if __name__ == '__main__':
    test02(0.5)