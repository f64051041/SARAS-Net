import torch
import torch.nn as nn
import torch.nn.functional as F
#from torchvision import models
#from base import BaseModel
#from utils.helpers import initialize_weights
from itertools import chain
#from swin_transformer import SwinTransformer
from einops import rearrange
from torchvision.models.utils import load_state_dict_from_url

GlobalAvgPool2D = lambda: nn.AdaptiveAvgPool2d(1)

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
}

class Cross_transformer_backbone(nn.Module):
    def __init__(self, in_channels = 48):
        super(Cross_transformer_backbone, self).__init__()
        
        self.to_key = nn.Linear(in_channels * 2, in_channels, bias=False)
        self.to_value = nn.Linear(in_channels * 2, in_channels, bias=False)
        self.softmax = nn.Softmax(dim=-1)

        self.gamma_cam_lay3 = nn.Parameter(torch.zeros(1))
        self.cam_layer0 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU()
        )
        self.cam_layer1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )
        self.cam_layer2 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )
        self.mlp = nn.Sequential(
            nn.Conv2d(in_channels*2, in_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, input_feature, features):
        Query_features = input_feature
        Query_features = self.cam_layer0(Query_features)       
        key_features = self.cam_layer1(features)
        value_features = self.cam_layer2(features)
        
        QK = torch.einsum("nlhd,nshd->nlsh", Query_features, key_features)
        softmax_temp = 1. / Query_features.size(3)**.5
        A = torch.softmax(softmax_temp * QK, dim=2)
        queried_values = torch.einsum("nlsh,nshd->nlhd", A, value_features).contiguous()
        message = self.mlp(torch.cat([input_feature, queried_values], dim=1))
        
        return input_feature + message

class Cross_transformer(nn.Module):
    def __init__(self, in_channels = 48):
        super(Cross_transformer, self).__init__()
        self.fa = nn.Linear(in_channels , in_channels, bias=False)
        self.fb = nn.Linear(in_channels, in_channels, bias=False)
        self.fc = nn.Linear(in_channels , in_channels, bias=False)
        self.fd = nn.Linear(in_channels, in_channels, bias=False)
        self.softmax = nn.Softmax(dim=-1)
        self.to_out = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )
        self.gamma_cam_lay3 = nn.Parameter(torch.zeros(1))
        self.fuse = nn.Sequential(
            nn.Conv2d(in_channels * 4, in_channels, kernel_size=1, padding=0),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )
        
    
    def attention_layer(self, q, k, v, m_batchsize, C, height, width):
        k = k.permute(0, 2, 1)
        energy = torch.bmm(q, k)
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy)-energy
        attention = self.softmax(energy_new)
        out = torch.bmm(attention, v)
        out = out.view(m_batchsize, C, height, width)
        
        return out
        
        
    def forward(self, input_feature, features):    
        fa = input_feature
        fb = features[0]
        fc = features[1]
        fd = features[2]
        

        m_batchsize, C, height, width = fa.size()
        fa = self.fa(fa.view(m_batchsize, C, -1).permute(0, 2, 1)).permute(0, 2, 1)
        fb = self.fb(fb.view(m_batchsize, C, -1).permute(0, 2, 1)).permute(0, 2, 1)
        fc = self.fc(fc.view(m_batchsize, C, -1).permute(0, 2, 1)).permute(0, 2, 1)
        fd = self.fd(fd.view(m_batchsize, C, -1).permute(0, 2, 1)).permute(0, 2, 1)
        
        
        qkv_1 = self.attention_layer(fa, fa, fa, m_batchsize, C, height, width)
        qkv_2 = self.attention_layer(fa, fb, fb, m_batchsize, C, height, width)  
        qkv_3 = self.attention_layer(fa, fc, fc, m_batchsize, C, height, width)
        qkv_4 = self.attention_layer(fa, fd, fd, m_batchsize, C, height, width)
        
        atten = self.fuse(torch.cat((qkv_1, qkv_2, qkv_3, qkv_4), dim = 1))
              

        out = self.gamma_cam_lay3 * atten + input_feature

        out = self.to_out(out)
        
        return out


class SceneRelation(nn.Module):
    def __init__(self,
                 in_channels,
                 channel_list,
                 out_channels,
                 scale_aware_proj=True):
        super(SceneRelation, self).__init__()
        self.scale_aware_proj = scale_aware_proj

        if scale_aware_proj:
            self.scene_encoder = nn.ModuleList(
                [nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, 1),
                    nn.ReLU(True),
                    nn.Conv2d(out_channels, out_channels, 1),
                ) for _ in range(len(channel_list))]
            )
        else:
            # 2mlp
            self.scene_encoder = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1),
                nn.ReLU(True),
                nn.Conv2d(out_channels, out_channels, 1),
            )
        self.content_encoders = nn.ModuleList()
        self.feature_reencoders = nn.ModuleList()
        for c in channel_list:
            self.content_encoders.append(
                nn.Sequential(
                    nn.Conv2d(c, out_channels, 1),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(True)
                )
            )
            self.feature_reencoders.append(
                nn.Sequential(
                    nn.Conv2d(c, out_channels, 1),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(True)
                )
            )

        self.normalizer = nn.Sigmoid()
        
        

    def forward(self, scene_feature, features: list):
        content_feats = [c_en(p_feat) for c_en, p_feat in zip(self.content_encoders, features)]

        scene_feats = [op(scene_feature) for op in self.scene_encoder]
        relations = [self.normalizer(sf) * cf for sf, cf in
                         zip(scene_feats, content_feats)]

        
        return relations

class PSPModule(nn.Module):
    def __init__(self, in_channels, bin_sizes=[1, 2, 4, 6]):
        super(PSPModule, self).__init__()
        out_channels = in_channels // len(bin_sizes)
        self.stages = nn.ModuleList([self._make_stages(in_channels, out_channels, b_s) 
                                                        for b_s in bin_sizes])
        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_channels+(out_channels * len(bin_sizes)), in_channels, 
                                    kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1)
        )

    def _make_stages(self, in_channels, out_channels, bin_sz):
        conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        bn = nn.BatchNorm2d(out_channels)
        relu = nn.ReLU(inplace=True)
        return nn.Sequential(conv, bn, relu)
    
    def forward(self, features):
        h, w = features.size()[2], features.size()[3]
        pyramids = [features]

        pyramids.extend([F.interpolate(stage(features), size=(h, w), mode='bilinear', 
                                        align_corners=True) for stage in self.stages])
        output = self.bottleneck(torch.cat(pyramids, dim=1))
        return output



class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out    

class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None, strides=None):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.strides = strides
        if self.strides is None:
            self.strides = [2, 2, 2, 2, 2]

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=self.strides[0], padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=self.strides[1], padding=1)
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=self.strides[1], padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=self.strides[2],
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=self.strides[3],
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=self.strides[4],
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)
        self.channe1 = nn.Sequential(
            nn.Conv2d(256, 64, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1)
        )            
        self.channe2 = nn.Sequential(
            nn.Conv2d(512, 128, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1)
        ) 
        self.channe3 = nn.Sequential(
            nn.Conv2d(1024, 256, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1)
        ) 
        self.channe4 = nn.Sequential(
            nn.Conv2d(2048, 512, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1)
        ) 

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        x = self.conv1(x) 
        x = self.bn1(x)
        x0 = self.relu(x)
        x00 = self.maxpool(x0) 
        x1 = self.layer1(x00) 
        x2 = self.layer2(x1) 
        x3 = self.layer3(x2) 
        x4 = self.layer4(x3) 
        x1 = self.channe1(x1)
        x2 = self.channe2(x2)
        x3 = self.channe3(x3)
        x4 = self.channe4(x4)
        return [x1, x2, x3, x4] 

    def forward(self, x):
        return self._forward_impl(x)

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            dilation = 1
            # raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out    


def _resnet(arch, block, layers, pretrained, progress, **kwargs):
    model = ResNet(block, layers, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch],
                                              progress=progress)
        model.load_state_dict(state_dict, strict=False)
    return model    

def resnet18(pretrained=False, progress=True, **kwargs):
    return _resnet('resnet18', BasicBlock, [2, 2, 2, 2], pretrained, progress,
                   **kwargs)    

def resnet50(pretrained=False, progress=True, **kwargs):
    return _resnet('resnet50', Bottleneck, [3, 4, 6, 3], pretrained, progress,
                   **kwargs)

class Change_detection(nn.Module):
    # Implementing only the object path
    def __init__(self, num_classes=2, use_aux=True, fpn_out=48, freeze_bn=False, **_):
        super(Change_detection, self).__init__()

        f_channels = [64, 128, 256, 512]

        # CNN-backbone
        self.resnet = resnet50(pretrained=True, replace_stride_with_dilation=[False,True,True])       
        self.PPN = PSPModule(f_channels[-1])


        
        # Relation-aware
        self.Cross_transformer_backbone_a3 =  Cross_transformer_backbone(in_channels = f_channels[3])
        self.Cross_transformer_backbone_a2 =  Cross_transformer_backbone(in_channels = f_channels[2])
        self.Cross_transformer_backbone_a1 =  Cross_transformer_backbone(in_channels = f_channels[1])
        self.Cross_transformer_backbone_a0 =  Cross_transformer_backbone(in_channels = f_channels[0])
        self.Cross_transformer_backbone_a33 =  Cross_transformer_backbone(in_channels = f_channels[3])
        self.Cross_transformer_backbone_a22 =  Cross_transformer_backbone(in_channels = f_channels[2])
        self.Cross_transformer_backbone_a11 =  Cross_transformer_backbone(in_channels = f_channels[1])
        self.Cross_transformer_backbone_a00 =  Cross_transformer_backbone(in_channels = f_channels[0])
                
        self.Cross_transformer_backbone_b3 =  Cross_transformer_backbone(in_channels = f_channels[3])
        self.Cross_transformer_backbone_b2 =  Cross_transformer_backbone(in_channels = f_channels[2])
        self.Cross_transformer_backbone_b1 =  Cross_transformer_backbone(in_channels = f_channels[1])
        self.Cross_transformer_backbone_b0 =  Cross_transformer_backbone(in_channels = f_channels[0])
        self.Cross_transformer_backbone_b33 =  Cross_transformer_backbone(in_channels = f_channels[3])
        self.Cross_transformer_backbone_b22 =  Cross_transformer_backbone(in_channels = f_channels[2])
        self.Cross_transformer_backbone_b11 =  Cross_transformer_backbone(in_channels = f_channels[1])
        self.Cross_transformer_backbone_b00 =  Cross_transformer_backbone(in_channels = f_channels[0])


        # Scale-aware
        self.sig = nn.Sigmoid()
        self.gap = GlobalAvgPool2D()
        self.sr1 = SceneRelation(in_channels = f_channels[3], channel_list = f_channels, out_channels = f_channels[3], scale_aware_proj=True)
        self.sr2 = SceneRelation(in_channels = f_channels[2], channel_list = f_channels, out_channels = f_channels[2], scale_aware_proj=True)
        self.sr3 = SceneRelation(in_channels = f_channels[1], channel_list = f_channels, out_channels = f_channels[1], scale_aware_proj=True)
        self.sr4 = SceneRelation(in_channels = f_channels[0], channel_list =f_channels, out_channels = f_channels[0], scale_aware_proj=True)


        # Cross transformer
        self.Cross_transformer1 =  Cross_transformer(in_channels = f_channels[3])
        self.Cross_transformer2 =  Cross_transformer(in_channels = f_channels[2])
        self.Cross_transformer3 =  Cross_transformer(in_channels = f_channels[1])
        self.Cross_transformer4 =  Cross_transformer(in_channels = f_channels[0])


        # Generate change map
        self.conv_fusion = nn.Sequential(
            nn.Conv2d(960 , fpn_out, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(fpn_out),
            nn.ReLU(inplace=True)
        )
        
        self.output_fill = nn.Sequential(
            nn.ConvTranspose2d(fpn_out , fpn_out, kernel_size=2, stride = 2, bias=False),
            nn.BatchNorm2d(fpn_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(fpn_out, num_classes, kernel_size=3, padding=1)
        )


    def forward(self, x1, x2):
        input_size = (x1.size()[-2], x1.size()[-1])

        # CNN-backbone
        features1 = self.resnet(x1)
        features2 = self.resnet(x2)
        
        features, features11, features22= [], [],[]

        # Relation-aware
        for i in range(len(features1)):
            if i == 0:
                features11.append(self.Cross_transformer_backbone_a00(features1[i] , self.Cross_transformer_backbone_a0(features1[i], features2[i])))
                features22.append(self.Cross_transformer_backbone_b00(features2[i], self.Cross_transformer_backbone_b0(features2[i], features1[i])))
            elif i == 1:
                features11.append(self.Cross_transformer_backbone_a11(features1[i] , self.Cross_transformer_backbone_a1(features1[i], features2[i])))
                features22.append(self.Cross_transformer_backbone_b11(features2[i], self.Cross_transformer_backbone_b1(features2[i], features1[i])))
            elif i == 2:    
                features11.append(self.Cross_transformer_backbone_a22(features1[i] , self.Cross_transformer_backbone_a2(features1[i], features2[i])))
                features22.append(self.Cross_transformer_backbone_b22(features2[i], self.Cross_transformer_backbone_b2(features2[i], features1[i])))
            elif i == 3:    
                features11.append(self.Cross_transformer_backbone_a33(features1[i] , self.Cross_transformer_backbone_a3(features1[i], features2[i])))
                features22.append(self.Cross_transformer_backbone_b33(features2[i], self.Cross_transformer_backbone_b3(features2[i], features1[i])))
          
        # The distance between features from two input images.
        for i in range(len(features1)):
            features.append(abs(features11[i] - features22[i])) 
        features[-1] = self.PPN(features[-1])


        # Scale-aware and cross transformer
        H, W = features[0].size(2), features[0].size(3)
        
        c6 = self.gap(features[-1])   
        c7 = self.gap(features[-2])    
        c8 = self.gap(features[-3])    
        c9 = self.gap(features[-4])   
        
        features1, features2, features3, features4 = [], [], [], []
        features1[:] = [F.interpolate(feature, size=(64, 64), mode='nearest') for feature in features[:]]
        list_3 = self.sr1(c6, features1) 
        fe3 = self.Cross_transformer1(list_3[-1], [list_3[-2], list_3[-3], list_3[-4]]) 
        
        features2[:] = [F.interpolate(feature, size=(64, 64), mode='nearest') for feature in features[:]]
        list_2 = self.sr2(c7, features2) 
        fe2 = self.Cross_transformer2(list_2[-2], [list_2[-1], list_2[-3], list_2[-4]]) 
        
        features3[:] = [F.interpolate(feature, size=(64, 64), mode='nearest') for feature in features[:]]
        list_1 = self.sr3(c8, features3) 
        fe1 = self.Cross_transformer3(list_1[-3], [list_1[-1], list_1[-2], list_1[-4]]) 
        
        features4[:] = [F.interpolate(feature, size=(128, 128), mode='nearest') for feature in features[:]]
        list_0 = self.sr4(c9, features4) 
        fe0 = self.Cross_transformer4(list_0[-4], [list_0[-1], list_0[-2], list_0[-3]]) 

        refined_fpn_feat_list = [fe3, fe2, fe1, fe0]
    
        # Upsampling 
        refined_fpn_feat_list[0] = F.interpolate(refined_fpn_feat_list[0], scale_factor=4, mode='nearest')
        refined_fpn_feat_list[1] = F.interpolate(refined_fpn_feat_list[1], scale_factor=4, mode='nearest')
        refined_fpn_feat_list[2] = F.interpolate(refined_fpn_feat_list[2], scale_factor=4, mode='nearest')
        refined_fpn_feat_list[3] = F.interpolate(refined_fpn_feat_list[3], scale_factor=2, mode='nearest')

        # Generate change map
        x = self.conv_fusion(torch.cat((refined_fpn_feat_list), dim=1))
        x = self.output_fill(x)


        return x


