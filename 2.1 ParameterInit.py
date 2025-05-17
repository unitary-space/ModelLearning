import torch
import torch.nn.functional as F
import torch.nn as nn

# 1. 均匀分布初始化参数
def test01():
    # 创建一个输入特征维度为 5，输出维度是 3 的模型
    linear = nn.Linear(5,3)
    nn.init.uniform_(linear.weight)

    print(linear.weight)
# 2. 固定初始化
def test02():
    linear = nn.Linear(5,3)
    nn.init.constant_(linear.weight, 5)
    print(linear.weight)

# 3. 全 0 初始化
def test03():
    # 偏置初始化为 0，权重不要为 0
    linear = nn.Linear(5,3)
    nn.init.zeros_(linear.weight)
    print(linear.weight)
# 4. 全 1 初始化
def test04():
    linear = nn.Linear(5,3)
    nn.init.ones_(linear.weight)
    print(linear.weight)
# 5. 随机初始化（正态分布）
def test05():
    linear = nn.Linear(5, 3)
    nn.init.normal_(linear.weight, mean=0,std=1)
    print(linear.weight)

# 6. Kaming 初始化
def test06():
    # 正态分布的初始化
    linear = nn.Linear(5, 3)
    nn.init.kaiming_normal_(linear.weight)
    print(linear.weight)

    linear = nn.Linear(5, 3)
    nn.init.kaiming_uniform_(linear.weight)
    print(linear.weight)
# 7. Xavier 初始化
def test07():
    linear = nn.Linear(5, 3)
    nn.init.xavier_normal_(linear.weight)
    print(linear.weight)

    linear = nn.Linear(5, 3)
    nn.init.xavier_uniform_(linear.weight)
    print(linear.weight)
if __name__ == '__main__':
    test07()