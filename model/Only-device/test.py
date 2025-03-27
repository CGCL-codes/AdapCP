import time
import torch
import torch.nn as nn
import torch.utils.benchmark as benchmark
import timeit
from torch.utils.data import Dataset, DataLoader
#import pandas as pd

# input0=torch.randn(6,3,4,4)
# targets0=torch.randn(6,6)
# input0=input0.cuda(cuda_num)
# targets0=targets0.cuda(cuda_num)
# my_input0=MyDataset(input0)
# data0=DataLoader(dataset=input0,batch_size=2,shuffle=True)
# targetdata0=DataLoader(dataset=targets0,batch_size=3,shuffle=True)



class Tudui(nn.Module):
    def __init__(self):
        super(Tudui, self).__init__()
        self.model = nn.Sequential(
            # nn.Conv2d(3, 32, 5, 1, 2),
            # nn.MaxPool2d(2),
            # nn.Conv2d(32, 32, 5, 1, 2),
            # nn.MaxPool2d(2),
            # nn.Conv2d(32, 64, 5, 1, 2),
            # nn.MaxPool2d(2),

            # nn.Flatten(),
            nn.Linear(1920, 12280),
        )

    def forward(self, x):
        x = self.model(x)
        return x

def init_weights(layer):
    # 如果为卷积层，使用正态分布初始化
    if type(layer) == nn.Conv2d:
        nn.init.normal_(layer.weight, mean=0, std=0.5)
    # 如果为全连接层，权重使用均匀分布初始化，偏置初始化为0.1
    elif type(layer) == nn.Linear:
        nn.init.uniform_(layer.weight, a=-0.1, b=0.1)
        nn.init.constant_(layer.bias, 0.1)

input0 = torch.randn(1920 )


tudui=Tudui()
tudui=tudui.apply(init_weights)
tudui.eval()
start=time.time()
result=tudui(input0)
end=time.time()

print(end-start)