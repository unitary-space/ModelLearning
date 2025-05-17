import torch
import torch.nn as nn
import pickle

'''
Pytorch 提供了两种保存模型的方法；
1. 直接序列化模型对象
2. 存储模型的全部参数
'''

class Model(nn.Module):
    def __init__(self, input_size, output_size):

        super(Model, self).__init__()
        self.linear1 = nn.Linear(input_size, output_size * 2)
        self.linear2 = nn.Linear(input_size * 2, output_size)

    def forward(self, inputs):

        inputs = self.linear1(inputs)
        outputs = self.linear2(inputs)
        return outputs

def test01():
    model = Model(128, 10)

    # 第一个参数：存储的模型
    # 第二个参数：存储的路径
    # 第三个参数：使用的模块
    # 第四个参数：存储的协议

    torch.save(model, 'model/test_model_save.bin', pickle_module=pickle, pickle_protocol=2)

def test02():

    # 第一个参数：存储的模型
    # 第二个参数：存储的路径
    # 第三个参数：使用的模块
    # 第四个参数：存储的协议
    model = torch.load('model/test_model_save.bin', map_location='cpu', pickle_module=pickle)

if __name__ == '__main__':
    test02()