---
title: iamge_transfer pytorch实现
date: 2017-12-24 03:23:59
tags: 机器学习
---
# image transfer pytorch 实现
前面有一篇博客讲了一些关于iamge transfer 的原理的，一共讲了5篇论文，这次就其中前两篇（这两篇是一样的）来实现一下，主要使用pytorch。  
[Image StyleTransfer Using Convolutional Neural Networks](https://pdfs.semanticscholar.org/7568/d13a82f7afa4be79f09c295940e48ec6db89.pdf)
## 模型回顾
先来回顾一下我们的模型，首先是我们的输入和输出，我们的输入是两个image，一个称为content，一个称为style，我们的输出是一张图片，这张图片应该有content的内容和style的纹理。而关于我们的CNN，我们使用的是已经训练好的VGG19的模型。下面是我们需要用的包，先进行导入。   
```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torch.autograd import Variable
import torchvision.transforms as transforms
import torchvision.models as models
import torch.utils.model_zoo as model_zoo
import math
import numpy as np
from PIL import Image
```
基本上都是pytorch的库，还有就是math numpy PIL之类
然后看一下能不能使用显卡加速
```python
use_cuda = torch.cuda.is_available()
print(use_cuda)
```
## 图片导入
这个问题的图片导入很简单，因为我们的训练只有两张图片。
```python
def load_data():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),
                             (0.229, 0.224, 0.225))
    ])  # 定义转换，将图片归一化。
    # 加载两张图片
    content_image = Image.open('./data/content.jpg')
    style_image = Image.open('./data/style.jpg')
    # 因为图片是 3*32*32的，但是pytorch接受的输入是4维的，所以要添加一个1 的维度，相当于变成了1*3*32*32 
    content_image = transform(content_image).unsqueeze(0)
    style_image = transform(style_image).unsqueeze(0)
    # 类型转换
    content_image = content_image.type(torch.FloatTensor)
    style_image = style_image.type(torch.FloatTensor)
    return content_image,style_image
```
## 定义网络
这部分还是比较难的，因为我们使用的是pre_trained的VGG19的网络，所以我们要去[pytorch的官网的VGG19](https://github.com/pytorch/vision/blob/master/torchvision/models/vgg.py)中找到VGG19的定义，并且将它稍稍改动一下。记住，不要使用自己写的vgg19，自己写的是没有办法导入官方的训练好的模型的。接下来说一下我们该怎样修改它官网的模型，根据论文的描述，我们需要拿到4个返回值，其中对于content，我们只需要它第四层池化后的结果，对与style，我们需要他们在每一层池化后的结果。所以，我们在取返回值的时候，我们并不是像正常的VGG19一样，取最后的output，而是将很多的中间结果都输出。下面是构建网络的代码（PS：代码很长，可以先不看）
```python
__all__ = [
    'VGG', 'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn',
    'vgg19_bn', 'vgg19',
]


model_urls = {
    'vgg11': 'https://download.pytorch.org/models/vgg11-bbd30ac9.pth',
    'vgg13': 'https://download.pytorch.org/models/vgg13-c768596a.pth',
    'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
    'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth',
    'vgg11_bn': 'https://download.pytorch.org/models/vgg11_bn-6002323d.pth',
    'vgg13_bn': 'https://download.pytorch.org/models/vgg13_bn-abd245e5.pth',
    'vgg16_bn': 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth',
    'vgg19_bn': 'https://download.pytorch.org/models/vgg19_bn-c79401a0.pth',
}


class VGG(nn.Module):

    def __init__(self, features, num_classes=1000):
        super(VGG, self).__init__()
        self.features = features
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )
        self._initialize_weights()
        self.content_return = ['26'] 
        self.style_return = ['3','8','17','26','35']

    def forward(self, x):
        content_outputs = []
        style_outputs = []
        for name, module in self.features._modules.items():
            x = module(x)
            if name in self.content_return:
                content_outputs += [x]
            if name in self.style_return:
                style_outputs += [x]
        return content_outputs, style_outputs

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


def vgg19(pretrained=False, **kwargs):
    """VGG 19-layer model (configuration "E")
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = VGG(make_layers(cfg['E']), **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['vgg19']))
    return model
```
代码比较长，但是注意，这基本上都是官方代码，都可以直接在[这里](https://github.com/pytorch/vision/blob/master/torchvision/models/vgg.py)找到，其中唯一不一样的是这个地方，就是关于forward的定义和init里面又多定义的两个参数。  
首先是官网上的forward，是这个样子的：
```python
def forward(self, x):
    x = self.features(x)       # 直接进入网络
    x = x.view(x.size(0), -1)  # 变成向量
    x = self.classifier(x)     # 全连接层
    return x 
```
而我们的forward是这样子的：
```python
def forward(self, x):
    content_outputs = []   # 关于内容的输出
    style_outputs = []     # 关于风格的输出
    for name, module in self.features._modules.items():
        x = module(x)
        if name in self.content_return:
            content_outputs += [x]
        if name in self.style_return:
            style_outputs += [x]
    return content_outputs, style_outputs
```
其中有很多需要注意的地方，首先是我先在init里面定义了我需要的返回的层的层数
```python
self.content_return = ['26'] 
self.style_return = ['3','8','17','26','35']
```
其次，在pytorch中，如果你不定义那些层的名字的话，那些层的名字将会是从0开始的数字，所以我上面定义的那些层数就是层的名字。当到达这些层的时候，我就把他们的结果放进我的list中，并在最后返回。
## Gram矩阵
参考在论文中的定义，Gram矩阵是这样的矩阵，首先假设我们在一个卷积层里面，我们有height* width * filter,我们用h，w，f来表示他们，我们的gram矩阵是一个f* f的矩阵，gram[i][j] = 第i个f和第j和f的内积。这里说内积是不准确的，因为每个f是一个h* w的矩阵，我们需要先将这些矩阵变成一个行向量，在进行内积。下面是gram的实现
```python
def gram(image):
    _,c,h,w = image.size()
    temp = image.view(c, h*w)
    return torch.mm(temp, temp.t()) 
```
代码很短，注意我们的图片在pytorch中是四维的，即1* c* h* w，这里的c是channel的意思和上面说的f是一样的。所以在第一行会有一个_，_就是1。之后直接view一下，相乘就可以了。
## 训练
终于到达了最重要的一环。训练，中间有些需要解释一下，首先是优化方法，在论文中，优化方式使用的是L-BFGS，但是这个十分占用空间，在我的GTX960m（4G显存）上跑了几个epoch就炸了，所以我选择了Adam作为优化方法。其次是关于loss的计算，这里采用的是欧式距离，但是在我使用官方库里的计算欧式距离的函数的时候出现了一些问题，所以，我最后采用了手动计算它的loss，毕竟根据pytorch的自动求导的功能，自己设计的函数也是可以工作的。最后介绍一下参数，content_image,style_image就不说了，alpha 和beta是用来调节content_loss和style_loss的比例的，m_lr是学习速率，epoch是迭代次数。
```python
def train(content_image, style_image, alpha, beta, m_lr,epoch):
    # 获得网络，True表示导入已经训练好的模型
    net = vgg19(True)
    print('model ok')
    # 是否使用显卡加速
    if use_cuda:
        content_image = content_image.cuda()
        style_image = style_image.cuda()
        net = net.cuda()
    # 根据论文，这里选择第四层的输出作为内容的部分的优化目标
    # 根据论文，这里采用每一层的gram矩阵作为优化的目标
    # 跟据论文，采用噪声或者content或者style优化是一样的,这里使用内容图像进行初始化
    temp = torch.rand(1,3,300,300)
    temp = temp.cuda()
    # 使用随机
    # input_image = Variable(temp.clone(), requires_grad = True)    
    # 使用content
    input_image = Variable(content_image.clone(), requires_grad = True)
    # 使用style
    # input_image = Variable(style_image.clone(), requires_grad = True)、
    # 获得优化的目标
    content_aim,_ = net(Variable(content_image,requires_grad = True))
    _,style_aim = net(Variable(style_image, requires_grad = True))
    # 这里使用Adam进行优化
    optimizer = optim.Adam([input_image],lr = m_lr )
    for i in range(epoch):
        # 初始化梯度
        optimizer.zero_grad()
        # 得到output
        predict_content, predict_style = net(input_image)
        # 求得content_loss
        content_loss = alpha * torch.mean((predict_content[0].data - content_aim[0].data)**2)
        # 求得style_loss
        style_loss = 0
        for j in range(5):
            _, c, h, w = predict_style[j].size()
            # 计算gram矩阵的值
            input_gram = gram(predict_style[j])
            style_gram = gram(style_aim[j])
            style_loss += torch.mean((input_gram - style_gram)**2)/(c*h*w)
        style_loss *= beta
        # 加权求和，求得总的loss，并反向传播
        total_loss = content_loss + style_loss
        total_loss.backward(retain_graph = True)
        optimizer.step()
        # 每10次打印出loss
        if i%10 == 0:
            print(str(i) + 'loss: '+str(total_loss))
    # 返回最后的结果
    return input_image
```
## main
最后是main函数，这里还要说一点，关于迭代次数和学习速率等参数，我没有认真地去调他们，应该还有其他的参数方式，读者可以自己去调节，试试不同效果。
```python
if __name__ == '__main__':
    content_image, style_image = load_data()
    print('load over')
    image = train(content_image,style_image,1,1000,0.05,1000)
    # 正则化
    denorm = transforms.Normalize((-2.12, -2.04, -1.80), (4.37, 4.46, 4.44))
    # 不要忘了是4维的，去掉前面的那一个维度
    image = image.clone().cpu().squeeze()
    image = denorm(image.data).clamp_(0, 1)
    torchvision.utils.save_image(image, 'output.png')
```
## 总结
这次算是我面做的时间最长的一个论文复现了，一共看了5篇论文，参考了github上的两三个人的代码，（有人用tensorflow做的，也有人用pytorch做的）花了大概两周的时间（一周看论文和CNN相关知识，一周用来实现），其中踩的最大的坑就是如何既要保证能导入官方已经训练好的模型，又能在网络中做出修改，拿出需要的人那几层卷积的内容。因为如果你自己定义网络的话，会很容易拿出那几层，但就不能导入官方的训练好模型，用官方的模型的话，就要深入它的模型中，修改它的forward函数，但是它的forward中CNN是集成在一个feature中的，最后在pytorch的社区，就是[这里](https://discuss.pytorch.org/t/how-to-extract-features-of-an-image-from-a-trained-model/119/49)找到了修改forward的方法。完成了这篇论文的复现。完整的代码以及图片效果可以在我的[github](https://github.com/LOOKCC)上看到。