import torch
import torch.nn as nn
import torch.nn.functional as F

from mmseg.core import add_prefix
from mmseg.ops import resize
from .. import builder
from ..builder import SEGMENTORS
from .base import BaseSegmentor

import numpy as np
import cv2

from .lib.conv_layer import Conv, BNPReLU
from .lib.axial_atten import AA_kernel
from .lib.context_module import CFPModule
from .utils import   BiRAFPN, ReverseAttention, ConvUp

@SEGMENTORS.register_module()
class BiRAFormer(nn.Module):
    """Encoder Decoder segmentors.

    EncoderDecoder typically consists of backbone, decode_head, auxiliary_head.
    Note that auxiliary_head is only used for deep supervision during training,
    which could be dumped during inference.
    """

    def __init__(self,
                 backbone,
                 decode_head,
                 neck=None,
                 auxiliary_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 compound_coef=4,
                 num_classes=1,
                 neo=False,numrepeat = 4,bottleneck=True):
        super(BiRAFormer, self).__init__()
        self.backbone = builder.build_backbone(backbone)
        if neck is not None:
            self.neck = builder.build_neck(neck)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.num_classes = num_classes
        self.fpn_num_filters = [64, 88, 112, 160, 224, 288, 384, 384, 384]
        self.fpn_cell_repeats = [5, 5,5, 5, 5, 5,5, 5, 5]#[3, 4, 5, 6, 7, 7, 8, 8, 8]
        self.numrepeat = numrepeat + 1
        self.compound_coef = compound_coef
        conv_channel_coef = {
            # the channels of P3/P4/P5.
            0: [40, 112, 320],
            1: [40, 112, 320],
            2: [48, 120, 352],
            3: [48, 136, 384],
            4: [56, 160, 448],#448
            5: [64, 176, 512],
            6: [72, 200, 576],
            7: [72, 200, 576],
            8: [80, 224, 640],
        }
        
        self.bifpn = nn.Sequential(
            *[BiRAFPN(self.fpn_num_filters[self.compound_coef],
                    conv_channel_coef[compound_coef],
                    True if _ == 0 else False,
                    attention=True ,
                    use_p8=compound_coef > 7, neo=neo,bottleneck=bottleneck)
              for _ in range(self.numrepeat)])
        self.conv1 = Conv(128,conv_channel_coef[compound_coef][0],1,1,padding=0,bn_acti=True)#128
        self.conv2 = Conv(320,conv_channel_coef[compound_coef][1],1,1,padding=0,bn_acti=True)#320
        self.conv3 = Conv(512,conv_channel_coef[compound_coef][2],1,1,padding=0,bn_acti=True)#512
        self.head1 = Conv(self.fpn_num_filters[compound_coef],num_classes,1,1,padding=0,bn_acti=False)
        self.head2 = Conv(self.fpn_num_filters[compound_coef],num_classes,1,1,padding=0,bn_acti=False)
        self.head3 = Conv(self.fpn_num_filters[compound_coef],num_classes,1,1,padding=0,bn_acti=False)
        self.ConvUp = ConvUp(self.fpn_num_filters[compound_coef],self.fpn_num_filters[compound_coef],True)
        self.ConvUp1 = ConvUp(self.fpn_num_filters[compound_coef],self.fpn_num_filters[compound_coef],True)
        self.ConvUp3 = ConvUp(self.fpn_num_filters[compound_coef],self.fpn_num_filters[compound_coef],True,True)
        
        self.dropout = nn.Dropout2d(0.2)
        self.prehead = Conv(self.fpn_num_filters[compound_coef],self.fpn_num_filters[compound_coef],3,1,padding=1,bn_acti=True)
    def forward(self, x):
        if self.num_classes==3:
            segout = self.backbone(x)
            x1 = segout[0]  #  64x88x88 /4
            x2 = segout[1]  # 128x44x44 /8
            x3 = segout[2]  # 320x22x22  /16
            x4 = segout[3]  # 512x11x11 /32

            x2 = self.conv1(x2)
            x3 = self.conv2(x3)
            x4 = self.conv3(x4)
            p3,p4,p5,p6,p7 = self.bifpn([x2,x3,x4])
            p5 = self.ConvUp(p3)
            p4 = self.ConvUp1(p5)
            p3 = self.ConvUp3(p4)
            p3 = self.prehead(p3)
            #p3 = self.dropout(p3)

            p3 = self.head3(p3)
            p4 = self.head2(p4)
            p5 = self.head1(p5)

            
            lateral_map_2 = F.interpolate(p5,scale_factor=4,mode='bilinear')
            lateral_map_5 = p3 #F.interpolate(p3,scale_factor=8,mode='bilinear') 
            lateral_map_3 = F.interpolate(p4,scale_factor=2,mode='bilinear') 
            lateral_map_1 =  p3 #F.interpolate(p3, scale_factor=8, mode='bilinear')
        elif self.num_classes ==1:
            segout = self.backbone(x)
            x1 = segout[0]  #  64x88x88 /4
            x2 = segout[1]  # 128x44x44 /8
            x3 = segout[2]  # 320x22x22  /16
            x4 = segout[3]  # 512x11x11 /32
            x2 = self.conv1(x2)
            x3 = self.conv2(x3)
            x4 = self.conv3(x4)
            p3,p4,p5,p6,p7 = self.bifpn([x2,x3,x4])
            p3 = self.head3(p3)
            p4 = self.head2(p4)
            p5 = self.head1(p5)
            lateral_map_2 = F.interpolate(p5,scale_factor=32,mode='bilinear')
            lateral_map_5 = F.interpolate(p3,scale_factor=8,mode='bilinear') 
            lateral_map_3 = F.interpolate(p4,scale_factor=16,mode='bilinear') 
            lateral_map_1 = F.interpolate(p3, scale_factor=8, mode='bilinear')
        return lateral_map_5,lateral_map_3,lateral_map_2,lateral_map_1