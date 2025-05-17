"""
BN 是批量归一化函数
"""
import torch
import torch.nn as nn



if __name__ == '__main__':
    # 输入的形状： [batch_size, channel, height, width]
    inputs = torch.randint(0, 10, [2, 2, 3, 4], dtype=torch.float32)

    # num_features 为样本特征图的数量，通道数量
    # affine 为是否带有 gamma beta参数
    # eps 小常数
    bn = nn.BatchNorm2d(num_features=2, affine=False, eps=1e-5)
    result = bn(inputs)
    print(inputs)
    print("-" * 80)
    print(result)

    # 均值是每个样本对应通道的均值，方差是对应通道的方差