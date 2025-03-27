import torch
import torch.nn as nn
import time
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
#from torchsummary import  summary
#from thop import profile


class NIN(nn.Module):
    def __init__(self,num_classes):
        super().__init__()

        # 每个Sequential里的计算量得一致，或相差不多，才能将GPU的利用率达到90%以上
        # in:512*512*3
        self.seq1 = nn.Sequential(
            nn.Conv2d(3, 96, 11, 4),  # 输入通道，输出通道（96个filters），核大小，步长    NIN是串联多个卷积层和全连接层的小网络。其中全连接层由1*1卷积代替
            nn.BatchNorm2d(96),
            nn.ReLU(),
            # in:96*125*125
            nn.Conv2d(96, 96, 1),  # 计算量：
            nn.BatchNorm2d(96),
            nn.ReLU(),
            nn.Conv2d(96, 96, 1),
            nn.BatchNorm2d(96),
            nn.ReLU(),
            nn.MaxPool2d(3, 2)
        )

        # in:96*62*62
        self.seq2 = nn.Sequential(
            nn.Conv2d(96, 192, 5, 1, 0),
            nn.BatchNorm2d(192),
            nn.ReLU(),
            # in:192*62*62
            nn.Conv2d(192, 192, 1),  # 计算量：
            nn.BatchNorm2d(192),
            nn.ReLU(),
            nn.Conv2d(192, 192, 1),
            nn.BatchNorm2d(192),
            nn.ReLU(),
            nn.MaxPool2d(3, 2, 0)
        )

        # in:192*31*31
        self.seq3 = nn.Sequential(
            nn.Conv2d(192, 384, 3, 1, 0),
            nn.BatchNorm2d(384),
            nn.ReLU(),
            # in:384*31*31
            nn.Conv2d(384, 384, 1),  # 计算量：
            nn.BatchNorm2d(384),
            nn.ReLU(),
            nn.Conv2d(384, 384, 1),
            nn.BatchNorm2d(384),
            nn.ReLU(),
            nn.MaxPool2d(3, 2)
        )

        # in:384*15*15
        self.seq4 = nn.Sequential(
            nn.Conv2d(384, 768, 3, 1, 0),
            nn.BatchNorm2d(768),
            nn.ReLU(),
            # in:768*15*15
            nn.Conv2d(768, 768, 1),  # 计算量：
            nn.BatchNorm2d(768),
            nn.ReLU(),
            nn.Conv2d(768, num_classes, 1),  # 做num_classes分类，将输出特征层变为num_classes
            nn.BatchNorm2d(num_classes),
            nn.ReLU()
        )

        # 全局平均池化
        self.p = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        x = self.seq1(x)
        x = self.seq2(x)
        x = self.seq3(x)
        x = self.seq4(x)

        x = self.p(x)
        x = x.view(len(x), -1)
        return x

def init_weights(layer):
    # 如果为卷积层，使用正态分布初始化
    if type(layer) == nn.Conv2d:
        nn.init.normal_(layer.weight, mean=0, std=0.5)
    # 如果为全连接层，权重使用均匀分布初始化，偏置初始化为0.1
    elif type(layer) == nn.Linear:
        nn.init.uniform_(layer.weight, a=-0.1, b=0.1)
        nn.init.constant_(layer.bias, 0.1)

transforms=transforms.Compose([
    transforms.Resize([200,200]),
    transforms.ToTensor()
])

def load_dataset(batch_size):
    train_set = torchvision.datasets.CIFAR100(
        root="../data/cifar-100", train=True,
        download=True, transform=transforms
    )
    test_set = torchvision.datasets.CIFAR100(
        root="../data/cifar-100", train=False,
        download=True, transform=transforms
    )
    train_iter = torch.utils.data.DataLoader(
        train_set, batch_size=batch_size, shuffle=True, num_workers=4
    )
    test_iter = torch.utils.data.DataLoader(
        test_set, batch_size=batch_size, shuffle=True, num_workers=4
    )
    return train_iter, test_iter

def test(net, test_iter, device):

    net.eval()
    count=1
    with torch.no_grad():
        print("*************** test ***************")
        for X, y in test_iter:
            if count==1:
                start_time=time.time()
            X, y = X.to(device), y.to(device)
            output = net(X)
            if count==1:
                inference_time=time.time()-start_time
                print(inference_time)
            count=count+1
    return inference_time

BATCH_SIZE = 200
NUM_EPOCHS = 10
NUM_CLASSES = 100
LEARNING_RATE = 0.02
MOMENTUM = 0.9
WEIGHT_DECAY = 0.0005
NUM_PRINT = 100
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"



def main():
    net = NIN(NUM_CLASSES)
    net = net.to(DEVICE)
    #summary(net, input_size=(3, 224, 224))
    #dummy_input = torch.randn(1, 3, 224, 224)
    #flops, params = profile(net, (dummy_input,))
    #print('flops: ', flops / 1000000.0, 'params: ', params / 1000000.0)
    train_iter, test_iter = load_dataset(BATCH_SIZE)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        net.parameters(),
        lr=LEARNING_RATE,
        momentum=MOMENTUM,
        weight_decay=WEIGHT_DECAY,
        nesterov=True
    )
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.1)


    NIN_test=net.apply(init_weights)
    inference_time=test(NIN_test,test_iter,DEVICE)
    print(inference_time)

if __name__ == '__main__':
    main()
