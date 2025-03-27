import torch
import torch.nn as nn
import time
import torchvision
import torchvision.transforms as transforms
from tensorboard import summary
from thop import profile


class MobleNetV1(nn.Module):
    def __init__(self, num_classes):
        super(MobleNetV1, self).__init__()
        self.conv1 = self._conv_st(3, 32, 2)
        self.conv_dw1 = self._conv_dw(32, 64, 1)
        self.conv_dw2 = self._conv_dw(64, 128, 2)
        self.conv_dw3 = self._conv_dw(128, 128, 1)
        self.conv_dw4 = self._conv_dw(128, 256, 2)
        self.conv_dw5 = self._conv_dw(256, 256, 1)
        self.conv_dw6 = self._conv_dw(256, 512, 2)
        self.conv_dw_x5 = self._conv_x5(512, 512, 5)
        self.conv_dw7 = self._conv_dw(512, 1024, 2)
        self.conv_dw8 = self._conv_dw(1024, 1024, 1)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(1024, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv_dw1(x)
        x = self.conv_dw2(x)
        x = self.conv_dw3(x)
        x = self.conv_dw4(x)
        x = self.conv_dw5(x)
        x = self.conv_dw6(x)
        x = self.conv_dw_x5(x)
        x = self.conv_dw7(x)
        x = self.conv_dw8(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

    def _conv_x5(self, in_channel, out_channel, blocks):
        layers = []
        for i in range(blocks):
            layers.append(self._conv_dw(in_channel, out_channel, 1))
        return nn.Sequential(*layers)

    def _conv_st(self, in_channels, out_channels, stride):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def _conv_dw(self, in_channels, out_channels, stride):
        return nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=stride, padding=1, groups=in_channels, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )





import torch.nn as nn
def mobleNetV1(num_classes):
    return MobleNetV1(num_classes=num_classes)





#define the initial function to init the layer's parameters for the network
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


# def train(net, train_iter, criterion, optimizer, num_epochs, device, num_print, lr_scheduler=None, test_iter=None):
#     net.train()
#     record_train = list()
#     record_test = list()
#
#     for epoch in range(num_epochs):
#         print("========== epoch: [{}/{}] ==========".format(epoch + 1, num_epochs))
#         total, correct, train_loss = 0, 0, 0
#         start = time.time()
#
#         for i, (X, y) in enumerate(train_iter):
#             X, y = X.to(device), y.to(device)
#             output = net(X)
#             loss = criterion(output, y)
#
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()
#
#             train_loss += loss.item()
#             total += y.size(0)
#             correct += (output.argmax(dim=1) == y).sum().item()
#             train_acc = 100.0 * correct / total
#
#             if (i + 1) % num_print == 0:
#                 print("step: [{}/{}], train_loss: {:.3f} | train_acc: {:6.3f}% | lr: {:.6f}" \
#                     .format(i + 1, len(train_iter), train_loss / (i + 1), \
#                             train_acc, get_cur_lr(optimizer)))
#
#
#         if lr_scheduler is not None:
#             lr_scheduler.step()
#
#         print("--- cost time: {:.4f}s ---".format(time.time() - start))
#
#         # if test_iter is not None:
#         #     record_test.append(test(net, test_iter, criterion, device))
#         # record_train.append(train_acc)
#
#     return record_train, record_test, net


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


def get_cur_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


# def learning_curve(record_train, record_test=None):
#     plt.style.use("ggplot")
#
#     plt.plot(range(1, len(record_train) + 1), record_train, label="train acc")
#     if record_test is not None:
#         plt.plot(range(1, len(record_test) + 1), record_test, label="test acc")
#
#     plt.legend(loc=4)
#     plt.title("learning curve")
#     plt.xticks(range(0, len(record_train) + 1, 5))
#     plt.yticks(range(0, 101, 5))
#     plt.xlabel("epoch")
#     plt.ylabel("accuracy")
#
#     plt.show()


import torch.optim as optim


BATCH_SIZE = 200
NUM_EPOCHS = 10
NUM_CLASSES = 100
LEARNING_RATE = 0.02
MOMENTUM = 0.9
WEIGHT_DECAY = 0.0005
NUM_PRINT = 100
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def main():
    net = mobleNetV1(NUM_CLASSES)
    net = net.to(DEVICE)
    summary(net, input_size=(3, 224, 224))
    dummy_input = torch.randn(1, 3, 32, 32)
    flops, params = profile(net, (dummy_input,))

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

    # record_train, record_test ,Mobilenet_test= train(net, train_iter, criterion, optimizer, \
    #       NUM_EPOCHS, DEVICE, NUM_PRINT, lr_scheduler, test_iter)
    Mobilenet_test=net.apply(init_weights)
    inference_time=test(Mobilenet_test,test_iter,DEVICE)
    print(inference_time)

if __name__ == '__main__':
    main()

