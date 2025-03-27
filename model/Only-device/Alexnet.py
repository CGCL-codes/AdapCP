import torch
import torch.nn as nn
import time
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
from torchsummary import summary

class AlexNet(nn.Module):
    def __init__(self, num_classes):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(256 , 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 256)
        x = self.classifier(x)
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
    transforms.Resize([224,224]),
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
    net = AlexNet(NUM_CLASSES)
    net = net.to(DEVICE)
    summary(net, input_size=(3, 224, 224))
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


    VGG16_test=net.apply(init_weights)
    inference_time=test(VGG16_test,test_iter,DEVICE)
    print(inference_time)

if __name__ == '__main__':
    main()
