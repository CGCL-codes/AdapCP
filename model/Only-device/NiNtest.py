import torch
from torch import nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from torchsummary import  summary
from thop import profile

def nin_block(in_channels, out_channels, kernel_size, strides, padding):
    # 定义NiN块
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, strides, padding),
        nn.ReLU(),
        nn.Conv2d(out_channels, out_channels, kernel_size=1), nn.ReLU(),
        nn.Conv2d(out_channels, out_channels, kernel_size=1), nn.ReLU())

# 定义NiN网络
net = nn.Sequential(
    nin_block(3, 96, kernel_size=11, strides=4, padding=0),
    nn.MaxPool2d(3, stride=2),
    nin_block(96, 256, kernel_size=5, strides=1, padding=2),
    nn.MaxPool2d(3, stride=2),
    nin_block(256, 384, kernel_size=3, strides=1, padding=1),
    nn.MaxPool2d(3, stride=2),
    nn.Dropout(0.5),
    nin_block(384, 10, kernel_size=3, strides=1, padding=1),
    nn.AdaptiveAvgPool2d((1, 1)),
    nn.Flatten())

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
def main():
    net.to(DEVICE)
    summary(net, input_size=(3, 224, 224))
    dummy_input = torch.randn(1, 3,224, 224)
    flops, params = profile(net, (dummy_input,))
    print('flops: ', flops / 1000000.0, 'params: ', params / 1000000.0)
if __name__ == '__main__':
    main()