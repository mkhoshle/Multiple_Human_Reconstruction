import sys, os
root_dir = os.path.join(os.path.dirname(__file__),'..')
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)
import torch
import torch.nn as nn
from config import args

from models.modelv1 import HOBJ as HOBJv1
from models.backbone import build_backbone
# from lib.models.hrnet_32 import HigherResolutionNet
# from models.resnet_50 import ResNet_50


Heads = {1: HOBJv1}

    
def build_model():
    backbone = build_backbone(args)
    # backbone = HigherResolutionNet()
        
    if args().model_version in Heads:
        HOBJ = Heads[args().model_version]
    else:
        raise NotImplementedError("Head is not recognized")
    model = HOBJ(backbone=backbone)
    
    return model


# def build_teacher_model(backbone='hrnet', head=1):
#     model = Heads[head](backbone=Backbones[backbone]())
#     return model