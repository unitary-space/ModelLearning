import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset


# 1. 数据类的构建
class SampleDataset(Dataset):

    def __init__(self, x, y):
        """初始化"""
        self.x = x
        self.y = y
        self.len = len(y)

    def __len__(self):
        """返回数据的总量"""
        return self.len

    def __getitem__(self, idx):
        """根据索引返回样本"""
        # 将 idx 限制在合理的范围内
        idx = min(max(idx, 0), self.len - 1)
        return self.x[idx], self.y[idx]


def test01():
    # 构建包含100个数据的数据集，每个样本有8个特征
    x = torch.randn(100, 8)
    y = torch.randint(0, 2, [x.size(0)])
    # 构建数据加载器的步骤：1.构建数据类，2.构建数据加载器
    sample_dataset = SampleDataset(x, y)
    print(sample_dataset[0])


# 2. 数据加载器的是使用
def test02():
    # 先构建数据对象
    x = torch.randn(100, 8)
    y = torch.randint(0, 2, [x.size(0)])
    sample_dataset = SampleDataset(x, y)
    # 构建数据加载器
    dataloader = DataLoader(sample_dataset, batch_size=4, shuffle=True)

    for x, y in dataloader:
        print(x)
        print(y)
        break


def test03():
    x = torch.randn(100, 8)
    y = torch.randint(0, 2, [x.size(0)])
    # 如果不想自己写数据类，可以直接调用 TensorDataset 类生成一个最简单的数据类
    sample_dataset = TensorDataset(x, y)
    data_loader = DataLoader(sample_dataset, batch_size=4, shuffle=True)

    for x, y in data_loader:
        print(x)
        print(y)

        break


if __name__ == '__main__':
    test03()
