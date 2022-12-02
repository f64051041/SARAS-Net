import torch
import torch.nn as nn
from torchvision.models import vgg16
import numpy as np
from torchvision.models.utils import load_state_dict_from_url
import torch.nn.functional as F



model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
    'resnext50_32x4d': 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
    'resnext101_32x8d': 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',
    'wide_resnet50_2': 'https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth',
    'wide_resnet101_2': 'https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth',
}

class vgg16_base(nn.Module):
    def __init__(self):
        super(vgg16_base,self).__init__()
        features = list(vgg16(pretrained=True).features)[:30]
        self.features = nn.ModuleList(features).eval()

    def forward(self,x):
        results = []
        for ii, model in enumerate(self.features):
            x = model(x)
            if ii in {3,8,15,22,29}:
                results.append(x)
        return results

class ChannelAttention(nn.Module):
    def __init__(self, in_channels, ratio = 8):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(in_channels,in_channels//ratio,1,bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_channels//ratio, in_channels,1,bias=False)
        self.sigmod = nn.Sigmoid()
    def forward(self,x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmod(out)

class SpatialAttention(nn.Module):
    def __init__(self):
        super(SpatialAttention,self).__init__()
        self.conv1 = nn.Conv2d(2,1,7,padding=3,bias=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        avg_out = torch.mean(x,dim=1,keepdim=True)
        max_out = torch.max(x,dim=1,keepdim=True,out=None)[0]

        x = torch.cat([avg_out,max_out],dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

def conv2d_bn(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels,out_channels,kernel_size=3,stride=1,padding=1),
        nn.PReLU(),
        nn.BatchNorm2d(out_channels),
        nn.Dropout(p=0.6),
    )



class UperNet(nn.Module):
    def __init__(self):
        super(UperNet, self).__init__()
        #self.t1_base = self.resnet = resnet18(pretrained=True, replace_stride_with_dilation=[False,True,True])
        self.t1_base =   vgg16_base()
        self.sa1 = SpatialAttention()
        self.sa2= SpatialAttention()
        self.sa3 = SpatialAttention()
        self.sa4 = SpatialAttention()
        self.sa5 = SpatialAttention()
        
        self.sigmoid = nn.Sigmoid()

        # branch1
        self.ca1 = ChannelAttention(in_channels=1024)
        self.bn_ca1 = nn.BatchNorm2d(1024)
        self.o1_conv1 = conv2d_bn(1024, 512)
        self.o1_conv2 = conv2d_bn(512, 512)
        self.bn_sa1 = nn.BatchNorm2d(512)
        self.o1_conv3 = nn.Conv2d(512, 1, 1)
        self.trans_conv1 = nn.ConvTranspose2d(512, 512, kernel_size=2, stride=2)

        # branch 2
        self.ca2 = ChannelAttention(in_channels=1536)
        self.bn_ca2 = nn.BatchNorm2d(1536)
        self.o2_conv1 = conv2d_bn(1536, 512)
        self.o2_conv2 = conv2d_bn(512, 256)
        self.o2_conv3 = conv2d_bn(256, 256)
        self.bn_sa2 = nn.BatchNorm2d(256)
        self.o2_conv4 = nn.Conv2d(256, 1, 1)
        self.trans_conv2 = nn.ConvTranspose2d(256, 256, kernel_size=2, stride=2)

        # branch 3
        self.ca3 = ChannelAttention(in_channels=768)
        self.o3_conv1 = conv2d_bn(768, 256)
        self.o3_conv2 = conv2d_bn(256, 128)
        self.o3_conv3 = conv2d_bn(128, 128)
        self.bn_sa3 = nn.BatchNorm2d(128)
        self.o3_conv4 = nn.Conv2d(128, 1, 1)
        self.trans_conv3 = nn.ConvTranspose2d(128, 128, kernel_size=2, stride=2)

        # branch 4
        self.ca4 = ChannelAttention(in_channels=384)
        self.o4_conv1 = conv2d_bn(384, 128)
        self.o4_conv2 = conv2d_bn(128, 64)
        self.o4_conv3 = conv2d_bn(64, 64)
        self.bn_sa4 = nn.BatchNorm2d(64)
        self.o4_conv4 = nn.Conv2d(64, 1, 1)
        self.trans_conv4 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)

        # branch 5
        self.ca5 = ChannelAttention(in_channels=192)
        self.o5_conv1 = conv2d_bn(192, 64)
        self.o5_conv2 = conv2d_bn(64, 32)
        self.o5_conv3 = conv2d_bn(32, 16)
        self.bn_sa5 = nn.BatchNorm2d(16)
        self.o5_conv4 = nn.Conv2d(16, 2, 1)

    def forward(self,t1_input,t2_input):
        t1_list = self.t1_base(t1_input)
        t2_list = self.t1_base(t2_input)
#         print("t1_list", t1_list.shape)
        t1_f_l3,t1_f_l8,t1_f_l15,t1_f_l22,t1_f_l29 = t1_list[0],t1_list[1],t1_list[2],t1_list[3],t1_list[4]
        t2_f_l3,t2_f_l8,t2_f_l15,t2_f_l22,t2_f_l29,= t2_list[0],t2_list[1],t2_list[2],t2_list[3],t2_list[4]

        
#         print("t1_list[4]", t1_list[4].shape)
#         print("t1_list[3]", t1_list[3].shape)
#         print("t1_list[2]", t1_list[2].shape)
#         print("t1_list[1]", t1_list[1].shape)
#         print("t1_list[0]", t1_list[0].shape)
#         t1_list[4] torch.Size([1, 512, 64, 64])  512, 32, 32
#         t1_list[3] torch.Size([1, 256, 64, 64])  512, 64, 64
#         t1_list[2] torch.Size([1, 128, 64, 64])  256, 128, 128
#         t1_list[1] torch.Size([1, 64, 128, 128]) 128, 256, 256
#         t1_list[0] torch.Size([1, 64, 256, 256]) 64, 512, 512
#         t1_list[0] torch.Size([1, 1024, 64, 64])
        x = torch.cat((t1_f_l29,t2_f_l29),dim=1) #512, 512
        #optional to use channel attention module in the first combined feature
        #在第一个深度特征叠加层之后可以选择使用或者不使用通道注意力模块
        # x = self.ca1(x) * x
        x = self.o1_conv1(x)
        x = self.o1_conv2(x)
        x = self.sa1(x) * x
        x = self.bn_sa1(x)
        
        branch_1_out = self.sigmoid(self.o1_conv3(x))
        x = self.trans_conv1(x)


        x = torch.cat((x,t1_f_l22,t2_f_l22),dim=1)
        x = self.ca2(x)*x
        #According to the amount of the training data, appropriately reduce the use of conv layers to prevent overfitting
        #根据训练数据的大小，适当减少conv层的使用来防止过拟合
        x = self.o2_conv1(x)
        x = self.o2_conv2(x)
        x = self.o2_conv3(x)
        x = self.sa2(x) *x
        x = self.bn_sa2(x)

        branch_2_out = self.sigmoid(self.o2_conv4(x))

        x = self.trans_conv2(x)
        x = torch.cat((x,t1_f_l15,t2_f_l15),dim=1)
        x = self.ca3(x)*x
        x = self.o3_conv1(x)
        x = self.o3_conv2(x)
        x = self.o3_conv3(x)
        x = self.sa3(x) *x
        x = self.bn_sa3(x)

        branch_3_out = self.sigmoid(self.o3_conv4(x))

        x = self.trans_conv3(x)
        x = torch.cat((x,t1_f_l8,t2_f_l8),dim=1)
        x = self.ca4(x)*x
        x = self.o4_conv1(x)
        x = self.o4_conv2(x)
        x = self.o4_conv3(x)
        x = self.sa4(x) *x
        x = self.bn_sa4(x)

        branch_4_out = self.sigmoid(self.o4_conv4(x))

        x = self.trans_conv4(x)
        x = torch.cat((x,t1_f_l3,t2_f_l3),dim=1)
        x = self.ca5(x)*x
        x = self.o5_conv1(x)
        x = self.o5_conv2(x)
        x = self.o5_conv3(x)
        x = self.sa5(x) *x
        x = self.bn_sa5(x)

        branch_5_out = self.sigmoid(self.o5_conv4(x))
        branch_5_out = F.interpolate(branch_5_out, size=t1_input.shape[2:], mode='bilinear',align_corners=True)
        return branch_5_out
