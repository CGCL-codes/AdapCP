import pickle

import torch
import torch.nn as nn
import time
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import socket



# ===================================================================
                    #客户端套接字
# ===================================================================

client=socket.socket(socket.AF_INET,socket.SOCK_STREAM)
client.connect(('192.168.137.1',6688))  #无论是客户端还是服务器端，connect中的ip，都是局域网中（不是以太网）服务器端的ip。因此注意要用ipconfig命令找，无线局域网中适配器本地连接中的ipv4地址。直接在网络设置中找到的ip地址是校园网通过有线网卡的分配的IP地址，不是局域网中服务器本身的ip地址

# data = Data(client_fx, labels, iter, self.local_ep, self.idx, len_batch)  客户端收发数据只需要这三句
# client.sendall(data)
# dfx = client.recv(50000)


# ===================================================================
                    #客户端模型
# ===================================================================
class AlexNet(nn.Module):
    def __init__(self, num_classes):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),    #0
            nn.ReLU(inplace=True),                                    #1
            nn.MaxPool2d(kernel_size=3, stride=2),                    #2
            nn.Conv2d(64, 192, kernel_size=5, padding=2),             #3
            nn.ReLU(inplace=True),                                    #4
            nn.MaxPool2d(kernel_size=3, stride=2),                    #5
            nn.Conv2d(192, 384, kernel_size=3, padding=1),            #6
            nn.ReLU(inplace=True),                                    #7
            nn.Conv2d(384, 256, kernel_size=3, padding=1),            #8
            nn.ReLU(inplace=True),                                    #9
            nn.Conv2d(256, 256, kernel_size=3, padding=1),            #10
            nn.ReLU(inplace=True),                                    #11
            nn.MaxPool2d(kernel_size=3, stride=2),                    #12
        )
        self.classifier = nn.Sequential(
            nn.Linear(256 , 4096),                                    #0
            nn.ReLU(inplace=True),                                    #1
            nn.Linear(4096, 4096),                                    #2
            nn.ReLU(inplace=True),                                    #3
            nn.Linear(4096, num_classes),                             #4
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
    transforms.Resize([64,64]),
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

def test(net, test_iter, layer,device):

    net.eval()
    count=True
    with torch.no_grad():
        print("*************** client inference ***************")
        for X, y in test_iter:
            X, y = X.to(device), y.to(device)
            client_inference_start_time=time.time()
            client_output = net.features[0:layer+1](X)                   #客户端模型执行部分
            client_sendData(client_output,client_inference_start_time,layer)


# ===================================================================
                    #数据处理
# ===================================================================

class Data(object):
    def __init__(self, client_output,client_inference_start_time,layer):
        self.client_output=client_output
        self.client_inference_start_time=client_inference_start_time
        self.layer=layer



def client_sendData(client_output,client_inference_start_time,laye):
    data=Data(client_output,client_inference_start_time,laye)
    # **序列化**——就是把内存里面的这些对象给变成一连串的字节描述的过程。
    # 序列化：把对象转换为字节序列的过程称为对象的序列化。
    str=pickle.dumps(data)

    # 先把str转换成长度为length的字节字符串，并发送
    client.send(len(str).to_bytes(length=6, byteorder='big'))
    #发送实际数据
    client.send(str)
    return False

def client_receiveData():
            lengthData=client.recv(6)
            if lengthData==b'quit':
                return lengthData
            length=int.from_bytes(lengthData, byteorder='big')
            b=bytes()
            count=0
            while True:
                value=client.recv(length)
                b=b+value
                count+=len(value)
                if count>=length:
                    break

            # 反序列化：把字节序列恢复为对象的过程称为对象的反序列化。
            data=pickle.loads(b)
            return data



BATCH_SIZE = 200
NUM_EPOCHS = 10
NUM_CLASSES = 100
LAYER=3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"



if __name__=="__main__":
    net = AlexNet(NUM_CLASSES)
    net = net.to(DEVICE)
    train_iter, test_iter = load_dataset(BATCH_SIZE)
    AlexNet_test=net.apply(init_weights)
    test(AlexNet_test,test_iter,LAYER,DEVICE)


