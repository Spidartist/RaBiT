import math
from turtle import forward

from torch import nn
import torch.nn.functional as F
import re
import math
import collections
from functools import partial
import torch
from torch.utils import model_zoo

class ConvUp(nn.Module):
    def __init__(self,in_channel=128,out_channel=128,activate = False,final=False):
        super(ConvUp, self).__init__()
        if final:
            self.final_up = nn.ConvTranspose2d(in_channel, out_channel, kernel_size=1, stride=2, output_padding=1)
        else:
            self.conv =  nn.Conv2d(in_channel, out_channel,
                              kernel_size=3, stride=1,
                              padding=1, bias=True)
        self.bn = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU()
        self.activate = activate
        self.final = final
    def forward(self, x,res=None):
        if self.final:
            x = self.final_up(x)
        else:
            x=  F.interpolate(x,scale_factor=2,mode='nearest')
            x = self.conv(x)
        x = self.bn(x)
        if self.activate:
            x = self.relu(x)
        if res != None:
            x = torch.cat((x,res),1)
        return x


class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x

def nms(dets, thresh):
    return nms_torch(dets[:, :4], dets[:, 4], thresh)


class SeparableConvBlock(nn.Module):
    """
    created by Zylo117
    """

    def __init__(self, in_channels, out_channels=None, norm=True, activation=False, onnx_export=False):
        super(SeparableConvBlock, self).__init__()
        if out_channels is None:
            out_channels = in_channels

        # Q: whether separate conv
        #  share bias between depthwise_conv and pointwise_conv
        #  or just pointwise_conv apply bias.
        # A: Confirmed, just pointwise_conv applies bias, depthwise_conv has no bias.

        self.depthwise_conv = Conv2dStaticSamePadding(in_channels, in_channels,
                                                      kernel_size=3, stride=1, groups=in_channels, bias=False)
        self.pointwise_conv = Conv2dStaticSamePadding(in_channels, out_channels, kernel_size=1, stride=1)

        self.norm = norm
        if self.norm:
            # Warning: pytorch momentum is different from tensorflow's, momentum_pytorch = 1 - momentum_tensorflow
            self.bn = nn.BatchNorm2d(num_features=out_channels, momentum=0.01, eps=1e-3)

        self.activation = activation
        if self.activation:
            self.swish = MemoryEfficientSwish() if not onnx_export else Swish()

    def forward(self, x):
        x = self.depthwise_conv(x)
        x = self.pointwise_conv(x)

        if self.norm:
            x = self.bn(x)

        if self.activation:
            x = self.swish(x)

        return x


class NeoReverseAttention(nn.Module):
    def __init__(self,channel,latenchannel = None) -> None:
        super(NeoReverseAttention,self).__init__()
        self.ra_conv1 = BasicConv2d(channel*3//2, channel//2, kernel_size=1)
        self.ra_conv2 = BasicConv2d(channel//2, channel//2, kernel_size=3, padding=1)     
        if latenchannel == None: 
            self.latenconv = BasicConv2d(channel, 3, kernel_size=1)
            self.stemconv = BasicConv2d(channel, channel//2, kernel_size=1)
        self.latenchannel = latenchannel
        self.ra_conv3 = BasicConv2d(channel//2, channel, kernel_size=1)
        
        self.channel =channel
        self.softmax = torch.nn.Softmax(1)
    def expand(self,inputs,res):
        x1 = torch.unsqueeze(inputs[:,0,:,:],1)
        x2 = torch.unsqueeze(inputs[:,1,:,:],1)
        x3 = torch.unsqueeze(inputs[:,2,:,:],1)
        x1 = x1.expand(-1,self.channel//2,-1,-1).mul(res)
        x2 = x2.expand(-1,self.channel//2,-1,-1).mul(res)
        x3 = x3.expand(-1,self.channel//2,-1,-1).mul(res)
        output = torch.cat((x1,x2,x3),1)
        return output
    def forward(self,inputs,latenmap ):
        if self.latenchannel == None:
            latenmap_ = self.latenconv(latenmap)
            inputs = self.stemconv(inputs)
        else:
            latenmap_ = latenmap
        x = -1*(self.softmax(latenmap_)) + 1
        x = self.expand(x,inputs)#x.expand(-1, self.channel, -1, -1).mul(inputs)
        
        x = self.ra_conv1(x)
        x = F.relu(self.ra_conv2(x))
        x =  F.relu(self.ra_conv3(x))
        if self.latenchannel == None:
            return x + latenmap
        else:
            return x + inputs        
class NeoReverseAttentionv2(nn.Module):
    def __init__(self,channel,latenchannel = None) -> None:
        super(NeoReverseAttentionv2,self).__init__()
        self.ra_conv1 = BasicConv2d(channel*3, channel, kernel_size=1)
        self.ra_conv2 = BasicConv2d(channel, channel, kernel_size=5, padding=2)     
        if latenchannel == None: 
            self.latenconv = BasicConv2d(channel, 3, kernel_size=1)
            self.stemconv = BasicConv2d(channel, channel, kernel_size=1)
        self.latenchannel = latenchannel
        self.ra_conv3 = BasicConv2d(channel, channel, kernel_size=1)
        
        self.channel =channel
        self.softmax = torch.nn.Softmax(1)
    def expand(self,inputs,res):
        x1 = torch.unsqueeze(inputs[:,0,:,:],1)
        x2 = torch.unsqueeze(inputs[:,1,:,:],1)
        x3 = torch.unsqueeze(inputs[:,2,:,:],1)
        x1 = x1.expand(-1,self.channel,-1,-1).mul(res)
        x2 = x2.expand(-1,self.channel,-1,-1).mul(res)
        x3 = x3.expand(-1,self.channel,-1,-1).mul(res)
        output = torch.cat((x1,x2,x3),1)
        return output
    def forward(self,inputs,latenmap ):
        if self.latenchannel == None:
            latenmap_ = self.latenconv(latenmap)
            inputs = self.stemconv(inputs)
        else:
            latenmap_ = latenmap
        x = -1*(self.softmax(latenmap_)) + 1
        x = self.expand(x,inputs)#x.expand(-1, self.channel, -1, -1).mul(inputs)
        
        x = self.ra_conv1(x)
        x = F.relu(self.ra_conv2(x))
        x =  F.relu(self.ra_conv3(x))
        if self.latenchannel == None:
            return x + latenmap
        else:
            return x + inputs              
class ReverseAttention(nn.Module):
    def __init__(self,channel,latenchannel = None,bottle_neck=True) -> None:
        super(ReverseAttention,self).__init__()
        if bottle_neck==False:
            self.ra_conv1 = BasicConv2d(channel, channel, kernel_size=1)
            self.ra_conv2 = BasicConv2d(channel, channel, kernel_size=5, padding=2)     
            if latenchannel == None: 
                self.latenconv = BasicConv2d(channel, 1, kernel_size=1)
            self.latenchannel = latenchannel
            self.ra_conv3 = BasicConv2d(channel, channel, kernel_size=1)
        else:
            self.ra_conv1 = BasicConv2d(channel, channel//2, kernel_size=1)
            self.ra_conv2 = BasicConv2d(channel//2, channel//2, kernel_size=3, padding=1)     
            if latenchannel == None: 
                self.latenconv = BasicConv2d(channel, 1, kernel_size=1)
            self.latenchannel = latenchannel
            self.ra_conv3 = BasicConv2d(channel//2, channel, kernel_size=1)
        self.channel =channel
    def forward(self,inputs,latenmap ):
        if self.latenchannel == None:
            latenmap_ = self.latenconv(latenmap)
        else:
            latenmap_ = latenmap
        x = -1*(torch.sigmoid(latenmap_)) + 1
        x = x.expand(-1, self.channel, -1, -1).mul(inputs)
        x = self.ra_conv1(x)
        x = F.relu(self.ra_conv2(x))
        x =  F.relu(self.ra_conv3(x))
        if self.latenchannel == None:
            return x + latenmap
        else:
            return x + inputs

class BiRAFPN(nn.Module):
    """
    modified by Thuan
    """

    def __init__(self, num_channels, conv_channels, first_time=False, epsilon=1e-4, onnx_export=True, attention=True,
                 use_p8=False,neo=False,bottleneck= True):
        """
        Args:
            num_channels:
            conv_channels:
            first_time: whether the input comes directly from the efficientnet,
                        if True, downchannel it first, and downsample P5 to generate P6 then P7
            epsilon: epsilon of fast weighted attention sum of BiFPN, not the BN's epsilon
            onnx_export: if True, use Swish instead of MemoryEfficientSwish
        """
        super(BiRAFPN, self).__init__()
        self.epsilon = epsilon
        self.use_p8 = use_p8

        # Conv layers
        self.conv6_up = SeparableConvBlock(num_channels, onnx_export=onnx_export)
        self.conv5_up = SeparableConvBlock(num_channels, onnx_export=onnx_export)
        self.conv4_up = SeparableConvBlock(num_channels, onnx_export=onnx_export)
        self.conv3_up = SeparableConvBlock(num_channels, onnx_export=onnx_export)
        self.conv4_down = SeparableConvBlock(num_channels, onnx_export=onnx_export)
        self.conv5_down = SeparableConvBlock(num_channels, onnx_export=onnx_export)
        self.conv6_down = SeparableConvBlock(num_channels, onnx_export=onnx_export)
        self.conv7_down = SeparableConvBlock(num_channels, onnx_export=onnx_export)
        if use_p8:
            self.conv7_up = SeparableConvBlock(num_channels, onnx_export=onnx_export)
            self.conv8_down = SeparableConvBlock(num_channels, onnx_export=onnx_export)

        # Feature scaling layers
        self.p6_upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.p5_upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.p4_upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.p3_upsample = nn.Upsample(scale_factor=2, mode='nearest')

        self.p4_downsample = MaxPool2dStaticSamePadding(3, 2)
        self.p5_downsample = MaxPool2dStaticSamePadding(3, 2)
        self.p6_downsample = MaxPool2dStaticSamePadding(3, 2)
        self.p7_downsample = MaxPool2dStaticSamePadding(3, 2)
        if use_p8:
            self.p7_upsample = nn.Upsample(scale_factor=2, mode='nearest')
            self.p8_downsample = MaxPool2dStaticSamePadding(3, 2)

        self.swish = MemoryEfficientSwish() if not onnx_export else Swish()

        self.first_time = first_time
        if self.first_time:
            self.p5_down_channel = nn.Sequential(
                Conv2dStaticSamePadding(conv_channels[2], num_channels, 1),
                nn.BatchNorm2d(num_channels, momentum=0.01, eps=1e-3),
            )
            self.p4_down_channel = nn.Sequential(
                Conv2dStaticSamePadding(conv_channels[1], num_channels, 1),
                nn.BatchNorm2d(num_channels, momentum=0.01, eps=1e-3),
            )
            self.p3_down_channel = nn.Sequential(
                Conv2dStaticSamePadding(conv_channels[0], num_channels, 1),
                nn.BatchNorm2d(num_channels, momentum=0.01, eps=1e-3),
            )

            self.p5_to_p6 = nn.Sequential(
                Conv2dStaticSamePadding(conv_channels[2], num_channels, 1),
                nn.BatchNorm2d(num_channels, momentum=0.01, eps=1e-3),
                MaxPool2dStaticSamePadding(3, 2)
            )
            self.p6_to_p7 = nn.Sequential(
                MaxPool2dStaticSamePadding(3, 2)
            )
            if use_p8:
                self.p7_to_p8 = nn.Sequential(
                    MaxPool2dStaticSamePadding(3, 2)
                )

            self.p4_down_channel_2 = nn.Sequential(
                Conv2dStaticSamePadding(conv_channels[1], num_channels, 1),
                nn.BatchNorm2d(num_channels, momentum=0.01, eps=1e-3),
            )
            self.p5_down_channel_2 = nn.Sequential(
                Conv2dStaticSamePadding(conv_channels[2], num_channels, 1),
                nn.BatchNorm2d(num_channels, momentum=0.01, eps=1e-3),
            )

        # Weight
        self.p6_w1 = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.p6_w1_relu = nn.ReLU()
        self.p5_w1 = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.p5_w1_relu = nn.ReLU()
        self.p4_w1 = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.p4_w1_relu = nn.ReLU()
        self.p3_w1 = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.p3_w1_relu = nn.ReLU()

        self.p4_w2 = nn.Parameter(torch.ones(3, dtype=torch.float32), requires_grad=True)
        self.p4_w2_relu = nn.ReLU()
        self.p5_w2 = nn.Parameter(torch.ones(3, dtype=torch.float32), requires_grad=True)
        self.p5_w2_relu = nn.ReLU()
        self.p6_w2 = nn.Parameter(torch.ones(3, dtype=torch.float32), requires_grad=True)
        self.p6_w2_relu = nn.ReLU()
        self.p7_w2 = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.p7_w2_relu = nn.ReLU()
        if neo:
            if bottleneck:
                self.RA_p7_6 = NeoReverseAttention(num_channels)
                self.RA_p6_5 = NeoReverseAttention(num_channels)
                self.RA_p5_4 = NeoReverseAttention(num_channels)
                self.RA_p4_3 = NeoReverseAttention(num_channels)
            else:
                print("using v2 neo")
                self.RA_p7_6 = NeoReverseAttentionv2(num_channels)
                self.RA_p6_5 = NeoReverseAttentionv2(num_channels)
                self.RA_p5_4 = NeoReverseAttentionv2(num_channels)
                self.RA_p4_3 = NeoReverseAttentionv2(num_channels)
        else:
            self.RA_p7_6 = ReverseAttention(num_channels,bottle_neck=bottleneck)
            self.RA_p6_5 = ReverseAttention(num_channels,bottle_neck=bottleneck)
            self.RA_p5_4 = ReverseAttention(num_channels,bottle_neck=bottleneck)
            self.RA_p4_3 = ReverseAttention(num_channels,bottle_neck=bottleneck)
        self.attention = attention

    def forward(self, inputs):
        """
        illustration of a minimal bifpn unit
            P7_0 -------------------------> P7_2 -------->
               |-------------|                ↑
                             ↓                |
            P6_0 ---------> P6_1 ---------> P6_2 -------->
               |-------------|--------------↑ ↑
                             ↓                |
            P5_0 ---------> P5_1 ---------> P5_2 -------->
               |-------------|--------------↑ ↑
                             ↓                |
            P4_0 ---------> P4_1 ---------> P4_2 -------->
               |-------------|--------------↑ ↑
                             |--------------↓ |
            P3_0 -------------------------> P3_2 -------->
        """

        # downsample channels using same-padding conv2d to target phase's if not the same
        # judge: same phase as target,
        # if same, pass;
        # elif earlier phase, downsample to target phase's by pooling
        # elif later phase, upsample to target phase's by nearest interpolation

        if self.attention:
            outs = self._forward_fast_attention(inputs)
        else:
            outs = self._forward(inputs)

        return outs

    def _forward_fast_attention(self, inputs):
        if self.first_time:
            p3, p4, p5 = inputs

            p6_in = self.p5_to_p6(p5)
            p7_in = self.p6_to_p7(p6_in)

            p3_in = self.p3_down_channel(p3)
            p4_in = self.p4_down_channel(p4)
            p5_in = self.p5_down_channel(p5)

        else:
            # P3_0, P4_0, P5_0, P6_0 and P7_0
            p3_in, p4_in, p5_in, p6_in, p7_in = inputs

        # P7_0 to P7_2

        # Weights for P6_0 and P7_0 to P6_1
        p6_w1 = self.p6_w1_relu(self.p6_w1)
        weight = p6_w1 / (torch.sum(p6_w1, dim=0) + self.epsilon)
        # Connections for P6_0 and P7_0 to P6_1 respectively
        temp_p6 = self.p6_upsample(p7_in)
        temp_p6 = self.RA_p7_6(p6_in,temp_p6)
        p6_up = self.conv6_up(self.swish(weight[0] * p6_in + weight[1] *temp_p6 ))

        # Weights for P5_0 and P6_1 to P5_1
        p5_w1 = self.p5_w1_relu(self.p5_w1)
        weight = p5_w1 / (torch.sum(p5_w1, dim=0) + self.epsilon)
        # Connections for P5_0 and P6_1 to P5_1 respectively

        temp_p5 = self.p5_upsample(p6_up)
        temp_p5 = self.RA_p6_5(p5_in,temp_p5)
        p5_up = self.conv5_up(self.swish(weight[0] * p5_in + weight[1] *temp_p5))

        # Weights for P4_0 and P5_1 to P4_1
        p4_w1 = self.p4_w1_relu(self.p4_w1)
        weight = p4_w1 / (torch.sum(p4_w1, dim=0) + self.epsilon)
        # Connections for P4_0 and P5_1 to P4_1 respectively

        temp_p4 = self.p4_upsample(p5_up)
        temp_p4 = self.RA_p5_4(p4_in,temp_p4)
        p4_up = self.conv4_up(self.swish(weight[0] * p4_in + weight[1] * temp_p4))

        # Weights for P3_0 and P4_1 to P3_2
        p3_w1 = self.p3_w1_relu(self.p3_w1)
        weight = p3_w1 / (torch.sum(p3_w1, dim=0) + self.epsilon)
        # Connections for P3_0 and P4_1 to P3_2 respectively

        temp_p3 =  self.p3_upsample(p4_up)
        temp_p3 = self.RA_p4_3(p3_in,temp_p3)
        p3_out = self.conv3_up(self.swish(weight[0] * p3_in + weight[1] * temp_p3))

        if self.first_time:
            p4_in = self.p4_down_channel_2(p4)
            p5_in = self.p5_down_channel_2(p5)

        # Weights for P4_0, P4_1 and P3_2 to P4_2
        p4_w2 = self.p4_w2_relu(self.p4_w2)
        weight = p4_w2 / (torch.sum(p4_w2, dim=0) + self.epsilon)
        # Connections for P4_0, P4_1 and P3_2 to P4_2 respectively
        p4_out = self.conv4_down(
            self.swish(weight[0] * p4_in + weight[1] * p4_up + weight[2] * self.p4_downsample(p3_out)))

        # Weights for P5_0, P5_1 and P4_2 to P5_2
        p5_w2 = self.p5_w2_relu(self.p5_w2)
        weight = p5_w2 / (torch.sum(p5_w2, dim=0) + self.epsilon)
        # Connections for P5_0, P5_1 and P4_2 to P5_2 respectively
        p5_out = self.conv5_down(
            self.swish(weight[0] * p5_in + weight[1] * p5_up + weight[2] * self.p5_downsample(p4_out)))

        # Weights for P6_0, P6_1 and P5_2 to P6_2
        p6_w2 = self.p6_w2_relu(self.p6_w2)
        weight = p6_w2 / (torch.sum(p6_w2, dim=0) + self.epsilon)
        # Connections for P6_0, P6_1 and P5_2 to P6_2 respectively
        p6_out = self.conv6_down(
            self.swish(weight[0] * p6_in + weight[1] * p6_up + weight[2] * self.p6_downsample(p5_out)))

        # Weights for P7_0 and P6_2 to P7_2
        p7_w2 = self.p7_w2_relu(self.p7_w2)
        weight = p7_w2 / (torch.sum(p7_w2, dim=0) + self.epsilon)
        # Connections for P7_0 and P6_2 to P7_2
        p7_out = self.conv7_down(self.swish(weight[0] * p7_in + weight[1] * self.p7_downsample(p6_out)))

        return p3_out, p4_out, p5_out, p6_out, p7_out

    def _forward(self, inputs):
        if self.first_time:
            p3, p4, p5 = inputs

            p6_in = self.p5_to_p6(p5)
            p7_in = self.p6_to_p7(p6_in)
            if self.use_p8:
                p8_in = self.p7_to_p8(p7_in)

            p3_in = self.p3_down_channel(p3)
            p4_in = self.p4_down_channel(p4)
            p5_in = self.p5_down_channel(p5)

        else:
            if self.use_p8:
                # P3_0, P4_0, P5_0, P6_0, P7_0 and P8_0
                p3_in, p4_in, p5_in, p6_in, p7_in, p8_in = inputs
            else:
                # P3_0, P4_0, P5_0, P6_0 and P7_0
                p3_in, p4_in, p5_in, p6_in, p7_in = inputs

        if self.use_p8:
            # P8_0 to P8_2

            # Connections for P7_0 and P8_0 to P7_1 respectively
            p7_up = self.conv7_up(self.swish(p7_in + self.p7_upsample(p8_in)))

            # Connections for P6_0 and P7_0 to P6_1 respectively
            p6_up = self.conv6_up(self.swish(p6_in + self.p6_upsample(p7_up)))
        else:
            # P7_0 to P7_2

            # Connections for P6_0 and P7_0 to P6_1 respectively
            p6_up = self.conv6_up(self.swish(p6_in + self.p6_upsample(p7_in)))

        # Connections for P5_0 and P6_1 to P5_1 respectively
        p5_up = self.conv5_up(self.swish(p5_in + self.p5_upsample(p6_up)))

        # Connections for P4_0 and P5_1 to P4_1 respectively
        p4_up = self.conv4_up(self.swish(p4_in + self.p4_upsample(p5_up)))

        # Connections for P3_0 and P4_1 to P3_2 respectively
        p3_out = self.conv3_up(self.swish(p3_in + self.p3_upsample(p4_up)))

        if self.first_time:
            p4_in = self.p4_down_channel_2(p4)
            p5_in = self.p5_down_channel_2(p5)

        # Connections for P4_0, P4_1 and P3_2 to P4_2 respectively
        p4_out = self.conv4_down(
            self.swish(p4_in + p4_up + self.p4_downsample(p3_out)))

        # Connections for P5_0, P5_1 and P4_2 to P5_2 respectively
        p5_out = self.conv5_down(
            self.swish(p5_in + p5_up + self.p5_downsample(p4_out)))

        # Connections for P6_0, P6_1 and P5_2 to P6_2 respectively
        p6_out = self.conv6_down(
            self.swish(p6_in + p6_up + self.p6_downsample(p5_out)))

        if self.use_p8:
            # Connections for P7_0, P7_1 and P6_2 to P7_2 respectively
            p7_out = self.conv7_down(
                self.swish(p7_in + p7_up + self.p7_downsample(p6_out)))

            # Connections for P8_0 and P7_2 to P8_2
            p8_out = self.conv8_down(self.swish(p8_in + self.p8_downsample(p7_out)))

            return p3_out, p4_out, p5_out, p6_out, p7_out, p8_out
        else:
            # Connections for P7_0 and P6_2 to P7_2
            p7_out = self.conv7_down(self.swish(p7_in + self.p7_downsample(p6_out)))

            return p3_out, p4_out, p5_out, p6_out, p7_out
class SwishImplementation(torch.autograd.Function):
    @staticmethod
    def forward(ctx, i):
        result = i * torch.sigmoid(i)
        ctx.save_for_backward(i)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        i = ctx.saved_variables[0]
        sigmoid_i = torch.sigmoid(i)
        return grad_output * (sigmoid_i * (1 + i * (1 - sigmoid_i)))


class MemoryEfficientSwish(nn.Module):
    def forward(self, x):
        return SwishImplementation.apply(x)


class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)#nn.functional.leaky_relu(x) #
class Conv2dStaticSamePadding(nn.Module):
    """
    created by Zylo117
    The real keras/tensorflow conv2d with same padding
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, bias=True, groups=1, dilation=1, **kwargs):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride,
                              bias=bias, groups=groups)
        self.stride = self.conv.stride
        self.kernel_size = self.conv.kernel_size
        self.dilation = self.conv.dilation

        if isinstance(self.stride, int):
            self.stride = [self.stride] * 2
        elif len(self.stride) == 1:
            self.stride = [self.stride[0]] * 2

        if isinstance(self.kernel_size, int):
            self.kernel_size = [self.kernel_size] * 2
        elif len(self.kernel_size) == 1:
            self.kernel_size = [self.kernel_size[0]] * 2

    def forward(self, x):
#         h, w = x.shape[2:]
        
#         extra_h = (math.ceil(w / self.stride[1]) - 1) * self.stride[1] - w + self.kernel_size[1]
#         extra_v = (math.ceil(h / self.stride[0]) - 1) * self.stride[0] - h + self.kernel_size[0]
        
#         left = extra_h // 2
#         right = extra_h - left
#         top = extra_v // 2
#         bottom = extra_v - top
        if self.kernel_size[0] == 1:
            x = F.pad(x,[0,0,0,0] )#[left, right, top, bottom])
        if self.kernel_size[0] == 3:
            x = F.pad(x,[1,1,1,1] )
        x = self.conv(x)
        #print("conv",self.stride,self.kernel_size,h,w,left, right, top, bottom ,"->",x.shape)
        return x


class MaxPool2dStaticSamePadding(nn.Module):
    """
    created by Zylo117
    The real keras/tensorflow MaxPool2d with same padding
    """

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.pool = nn.MaxPool2d(*args, **kwargs)
#         self.stride = self.pool.stride
#         self.kernel_size = self.pool.kernel_size

#         if isinstance(self.stride, int):
#             self.stride = [self.stride] * 2
#         elif len(self.stride) == 1:
#             self.stride = [self.stride[0]] * 2

#         if isinstance(self.kernel_size, int):
#             self.kernel_size = [self.kernel_size] * 2
#         elif len(self.kernel_size) == 1:
#             self.kernel_size = [self.kernel_size[0]] * 2

    def forward(self, x):
#         h, w = x.shape[2:]
        
#         extra_h = (math.ceil(w / self.stride[1]) - 1) * self.stride[1] - w + self.kernel_size[1]
#         extra_v = (math.ceil(h / self.stride[0]) - 1) * self.stride[0] - h + self.kernel_size[0]

#         left = extra_h // 2
#         right = extra_h - left
#         top = extra_v // 2
#         bottom = extra_v - top
        
        x = F.pad(x,[0,1,0,1] )#[left, right, top, bottom])

        x = self.pool(x)
      #  print("max-pool",h,w,left, right, top, bottom,"->",x.shape)
        return x