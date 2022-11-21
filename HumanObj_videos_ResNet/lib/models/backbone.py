# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Backbone modules.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from typing import Dict, List
from config import args
import torch
import torch.nn as nn
import torchvision.models.resnet as resnet
import torchvision.transforms.functional as F
import sys, os
root_dir = os.path.join(os.path.dirname(__file__),'..')
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)
from utils import BHWC_to_BCHW, copy_state_dict
from models.CoordConv import get_coord_maps
import config
from config import args
from models.hrnet_32 import HigherResolutionNet
from models.resnet_50 import ResNet_50

# from utils.misc import is_main_process, NestedTensor
from models.position_encoding import build_position_encoding
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
BN_MOMENTUM = 0.1




Backbones = {'hrnet': HigherResolutionNet, 'resnet': ResNet_50}

class Joiner(nn.Sequential):
    def __init__(self, backbone):
        super().__init__(backbone)
        self.backbone = backbone
        # self.position_embedding = position_embedding

    def forward(self, tensor):
        out = self.backbone(tensor)
           
        # position encoding        
        # pos = self.position_embedding(out)
        
        return out


def build_backbone(args):
    # position_embedding = build_position_encoding(args)
    train_backbone = args().lr_backbone > 0
    return_interm_layers = args().masks or (args().num_feature_levels > 1)
    
    if args().backbone in Backbones:
        backbone = Backbones[args().backbone]()
    elif args().backbone=="resnet_fpn_backbone":
        backbone = resnet_fpn_backbone('resnet50', pretrained=False, trainable_layers=5)
        if os.path.exists(args().resnet_pretrain):
            checkpoint = torch.load(args().resnet_pretrain)
            for key in list(checkpoint.keys()):
                if 'backbone.' in key:
                    checkpoint[key.replace('backbone.', '')] = checkpoint[key]
                    del checkpoint[key]
            backbone = backbone.load_state_dict(checkpoint)
        
    
    model = Joiner(backbone)
    return model
    


# class BackboneBase(nn.Module):

#     def __init__(self, backbone: nn.Module, train_backbone: bool,
#                  return_interm_layers: bool):
#         super().__init__()
#         for name, parameter in backbone.named_parameters():
#             if (not train_backbone
#                 or 'layer2' not in name
#                 and 'layer3' not in name
#                 and 'layer4' not in name):
#                 parameter.requires_grad_(False)
#         if return_interm_layers:
#             return_layers = {"layer1": "0", "layer2": "1", "layer3": "2", "layer4": "3"}
#             # return_layers = {"layer2": "0", "layer3": "1", "layer4": "2"}
#             self.strides = [4, 8, 16, 32]
#             self.num_channels = [256, 512, 1024, 2048]
#         else:
#             return_layers = {'layer3': "0"}
#             self.strides = [16]
#             self.num_channels = [1024]
#         self.body = IntermediateLayerGetter(backbone, return_layers=return_layers)

#     def forward(self, tensor_list: NestedTensor):
#         xs = self.body(tensor_list.tensors)
#         out: Dict[str, NestedTensor] = {}
#         for name, x in xs.items():
#             m = tensor_list.mask
#             assert m is not None
#             mask = F.interpolate(m[None].float(), size=x.shape[-2:]).to(torch.bool)[0]
#             out[name] = NestedTensor(x, mask)
#         return out


# class Backbone(BackboneBase):
#     """ResNet backbone with frozen BatchNorm."""
#     def __init__(self, name: str,
#                  train_backbone: bool,
#                  return_interm_layers: bool,
#                  dilation: bool):
#         norm_layer = FrozenBatchNorm2d
#         backbone = getattr(torchvision.models, name)(
#             replace_stride_with_dilation=[False, False, dilation],
#             pretrained=is_main_process(), norm_layer=norm_layer)
#         super().__init__(backbone, train_backbone,
#                          return_interm_layers)
#         if dilation:
#             self.strides[-1] = self.strides[-1] // 2
