---
title: pytorch cv 模板
date: 2018-01-30 03:10:49
tags: pytorch
---
# Pytorch 模板
pytorch在处理图像问题的时候是有一定的模板的，照着这些模板，可以很快速的写出一个cnn的模型，这次就写一个博客来存一下这个模板。
## utils
首先是关于data的处理，因为是cv问题，所以一般处理的都是图像，下面这个是pytorch的官网的的方式，在的官方的推荐里面，比如cifar10 的代码里面都是这样使用的。
```python
import torch
import torch.utils.data as Data
from torchvision import transforms,datasets
# 首先是定义自己的transform 可以对图片进行裁剪变换等等，相应的操作可以看他们的官网的文档，http://pytorch.org/docs/master/torchvision/transforms.html
# 这里特别要注意这要求是PIL导进去的image，其他的格式是不行的。
crop_transform = transforms.Compose([
    transforms.Scale(256),
    transforms.RandomCrop(224),
    transforms.ToTensor(),
])
# 之后定义自己的类，这个类继承 它的Dataset类，里面至少要重载三个的方法
class ImageLoader(Data.Dataset):
    # 初始化类
    def __init__(self, image_list, root_dir, transform):
        self.image_list = image_list
        self.root_dir = root_dir
        self.transform = transform
    # 获得长度，重载后就可以使用len() 这个方法
    def __len__(self):
        return len(self.image_list)
    # 最重要的一个类，这个类是用来迭代的，也就是说，从在这个类之后，就可以使用[] 操作了，同时也要在这个列里面实现你要对图片的各种操作，或者说是对数据的处理
    def __getitem__(self, idx):
        # 灰度 剪切 前面定义的transform 等等
        return data_set, label_set
# 最后是数据打包的操作，先定义所有的数据或者说是对数据的操作，然后进行batch打包
all_data = ImageLoader(image_list,dir,crop_transform)
    loader = Data.DataLoader(
        dataset=all_data,     
        batch_size = m_batch_size,     
        shuffle=True,              
        num_workers=2,             
    )
```
## net
接下来是网络部分，这部分有两个方法，第一种是清楚直观的方法，就是全部写出来，比如下面这样，是一个VGG19.
```py
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1_1 = nn.Conv2d(3, 64, 3, padding=1) 
        self.conv1_2 = nn.Conv2d(64, 64, 3, padding=1)
        self.pool1 = nn.MaxPool2d(2,2)
        self.conv2_1 = nn.Conv2d(64, 128, 3, padding=1) 
        self.conv2_2 = nn.Conv2d(128, 128, 3, padding=1)
        self.conv2_3 = nn.Conv2d(128, 128, 3, padding=1)  
        self.pool2 = nn.MaxPool2d(2,2)
        self.conv3_1 = nn.Conv2d(128, 256, 3, padding=1) 
        self.conv3_2 = nn.Conv2d(256, 256, 3, padding=1) 
        self.conv3_3 = nn.Conv2d(256, 256, 3, padding=1) 
        self.conv3_4 = nn.Conv2d(256, 256, 3, padding=1) 
        self.pool3 = nn.MaxPool2d(2,2)
        self.conv4_1 = nn.Conv2d(256, 512, 3, padding=1) 
        self.conv4_2 = nn.Conv2d(512, 512, 3, padding=1) 
        self.conv4_3 = nn.Conv2d(512, 512, 3, padding=1) 
        self.conv4_4 = nn.Conv2d(512, 512, 3, padding=1) 
        self.pool4 = nn.MaxPool2d(2,2)
        self.conv5_1 = nn.Conv2d(512, 512, 3, padding=1) 
        self.conv5_2 = nn.Conv2d(512, 512, 3, padding=1) 
        self.conv5_3 = nn.Conv2d(512, 512, 3, padding=1) 
        self.conv5_4 = nn.Conv2d(512, 512, 3, padding=1) 
        self.pool5 = nn.MaxPool2d(2,2)
        self.relu = nn.ReLU(inplace=True)
        self.classifier = nn.Linear(512,10)
 
    def forward(self, x):
        x = self.conv1_1(x)
        x = self.relu(x)
        x = self.conv1_2(x)
        x = self.relu(x)
        x = self.pool1(x)
        x = self.conv2_1(x)
        x = self.relu(x)
        x = self.conv2_2(x)
        x = self.relu(x)
        x = self.pool2(x)
        x = self.conv3_1(x)
        x = self.relu(x)
        x = self.conv3_2(x)
        x = self.relu(x)
        x = self.conv3_3(x)
        x = self.relu(x)
        x = self.conv3_4(x)
        x = self.relu(x)
        x = self.pool3(x)
        x = self.conv4_1(x)
        x = self.relu(x)
        x = self.conv4_2(x)
        x = self.relu(x)
        x = self.conv4_3(x)
        x = self.relu(x)
        x = self.conv4_4(x)
        x = self.relu(x)
        x = self.pool4(x)
        x = self.conv5_1(x)
        x = self.relu(x)
        x = self.conv5_2(x)
        x = self.relu(x)
        x = self.conv5_3(x)
        x = self.relu(x)
        x = self.conv5_4(x)
        x = self.relu(x)
        x = self.pool5(x)
        x = x.view(-1, 512)
        x = self.classifier(x)
        return x
```
但是像上面这样的话，又太过繁琐，还有就是像pytorch官网的那样的，这里是链接https://github.com/pytorch/vision/blob/master/torchvision/models/vgg.py
```py
class VGG(nn.Module):

    def __init__(self, features, num_classes=1000, init_weights=True):
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
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

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
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG(make_layers(cfg['E']), **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['vgg19']))
    return model

```
## train
最后就是train了，这个也是都十分固定的，基本就是导入数据，定义优化方法和网络，然后开始按照epoch，最重要的是在循环中的关于loss的定义。
```py
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import net
import loaddata

dir = './VOCdevkit/VOC2012/JPEGImages/'
learning_rate = 0.1
epoch = 1000
use_cuda = torch.cuda.is_available()
filenames = loaddata.getdir(0,10)
loader = loaddata.dataloader(filenames,dir,5)
print('load ok')
colornet = net.ColorNet()
if use_cuda:
    colornet = colornet.cuda() 
mes_loss = nn.MSELoss()
optimizer = optim.Adam(colornet.parameters(),lr = learning_rate)
for i in range(epoch):
    for batch_idx,(inputs,targets,origin) in enumerate(loader):
        if use_cuda:
            inputs,targets = inputs.cuda(), targets.cuda()
        optimizer.zero_grad()
        inputs = Variable(inputs)
        targets = Variable(targets)
        outputs = colornet(inputs)
        loss = mes_loss(outputs,targets) 
        loss.backward()
        optimizer.step()
    if i%10 == 0:
        print(str(i+1) + ' loss: '+str(loss))
```
## 总结 
有了这个模板之后，以后写起来就十分方便了。