import pickle

import torch
import torch.nn as nn
import time
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import socket



# ====================================================================================================
#                                  服务器套接字
# ====================================================================================================


# ====================================================================================================
#                                  Data transmission
# ====================================================================================================
class Data(object):   #server端的Data必须和client端的Data类同名。如果不一样，那么下边data = pickle.loads(b)时候，由于client客户端发送过来的Data对象，而server这边没有对应的Data类定义，就会报错找不到Data attribute。

    def __init__(self, client_output,trans_start_time):
        self.client_output=client_output
        self.trans_start_time=trans_start_time



def server_sendData(server,client_output,trans_start_time):
    data=Data(client_output,trans_start_time)
    # **序列化**——就是把内存里面的这些对象给变成一连串的字节描述的过程。
    # 序列化：把对象转换为字节序列的过程称为对象的序列化。
    str=pickle.dumps(data)

    # 把str转换成长度为length的字节字符串，并发送
    server.send(len(str).to_bytes(length=6, byteorder='big'))
    server.send(str)


def server_receiveData(model):
    while True:
        print(u'waiting for connect...')
        # 等待连接，一旦有客户端连接后，返回一个建立了连接后的套接字和连接的客户端的IP和端口元组
        connect, (host, port) = server.accept()
        print(u'the client %s:%s has connected.' % (host, port))
        while True:
            lengthData = connect.recv(6)
            print('lengthData:', lengthData)
            length = int.from_bytes(lengthData, byteorder='big')
            print('lengthData.int:',length)
            b = bytes()
            if length == 0:
                continue
            count=0
            while True:
                print("----------data accepting---------")
                value = connect.recv(length)
                b = b + value
                count += len(value)
                if count >= length:
                    break
            print("----------data accepted---------")
            # 反序列化：把字节序列恢复为对象的过程称为对象的反序列化。
            data = pickle.loads(b)
            test(model, data.client_output, data.layer, data.client_inference_start_time)




# ====================================================================================================
#                                  NN-model Program
# ====================================================================================================

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
            nn.Linear(256 , 4096),                                    #13
            nn.ReLU(inplace=True),                                    #14
            nn.Linear(4096, 4096),                                    #15
            nn.ReLU(inplace=True),                                    #16
            nn.Linear(4096, num_classes),                             #17
        )

    def forward(self, x,layer):
        x = self.features[layer+1:13](x)
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

def test(net, client_output,layer,client_inference_start_time):

    net.eval()

    with torch.no_grad():
        print("*************** server inference ***************")

        server_output = net(client_output,layer)                   #客户端模型执行部分
        total_inference_time=time.time()-client_inference_start_time
    print('total_inference_time:',total_inference_time)


BATCH_SIZE = 200
NUM_EPOCHS = 10
NUM_CLASSES = 100
LAYER=3
DEVICE = "cpu"



if __name__=="__main__":

    net = AlexNet(NUM_CLASSES)
    net = net.to(DEVICE)
    AlexNet_test=net.apply(init_weights)

    # 创建一个socket套接字，该套接字还没有建立连接
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setblocking(1)
    # 绑定监听端口，这里必须填本机的IP192.168.137.1，localhost和127.0.0.1是本机之间的进程通信使用的
    # 无论是客户端还是服务器端，connect中的ip，都是局域网中（不是以太网）服务器端的ip。因此注意要用ipconfig命令找，无线局域网中适配器本地连接中的ipv4地址。直接在网络设置中找到的ip地址是校园网通过有线网卡的分配的IP地址，不是局域网中服务器本身的ip地址
    server.bind(('192.168.137.1', 6688))
    # 开始监听，并设置最大连接数
    server.listen(1)

    # print(u'waiting for connect...')
    # # 等待连接，一旦有客户端连接后，返回一个建立了连接后的套接字和连接的客户端的IP和端口元组
    # connect, (host, port) = server.accept()
    # print(u'the client %s:%s has connected.' % (host, port))
    server_receiveData(AlexNet_test)




