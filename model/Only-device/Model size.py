import torch
from torchvision import models
from torchsummary import  summary
from thop import profile


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
net=models.MobileNetV2()
net = net.to(DEVICE)
summary(net, input_size=(3, 224, 224))
dummy_input = torch.randn(1, 3, 224, 224)
flops, params = profile(net, (dummy_input,))
print('flops: ', flops/1000000.0, 'params: ', params/1000000.0)