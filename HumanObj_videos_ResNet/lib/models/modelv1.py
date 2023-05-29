from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn

import sys, os
root_dir = os.path.join(os.path.dirname(__file__),'..')
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)
from models.base import Base
from models.CoordConv import get_coord_maps
from models.transformer import build_transformer

import config
from config import args
from loss_funcs import Loss
from maps_utils.result_parser import ResultParser
from utils.misc import NestedTensor, nested_tensor_from_tensor_list

from models.position_encoding import build_position_encoding

BN_MOMENTUM = 0.1


class HOBJ(Base):
    def __init__(self, backbone=None, **kwargs):
        super(HOBJ, self).__init__()
        print('Using HOBJ v1')    
        self.backbone = backbone
        self.backbone = self.backbone.cuda()
        self.position_embedding = build_position_encoding(args)
        self._result_parser = ResultParser()
        self._build_model()
        # self.input_proj = nn.Conv2d(256, 64, kernel_size=1)    # This is for Faster-RCNN
        # self.input_proj = nn.Conv2d(64, 64, kernel_size=8,stride=8)  # 16*16 heatmap
        # self.input_proj = nn.Conv2d(64, 16, kernel_size=3,stride=2,padding=1)
        # self.input_proj = nn.Conv2d(64, 64, kernel_size=3,stride=4,padding=1)  # 32 heatmap size
        self.pool = nn.MaxPool2d(kernel_size=4, stride=4)
        # self.input_proj = nn.Conv2d(64, 16, kernel_size=3,stride=2,padding=1)  # 64 heatmap size
        self.output_proj1 = nn.Conv2d(self.hidden_dim, self.output_cfg['NUM_CENTER_MAP'], kernel_size=1)
        self.output_proj2 = nn.Conv2d(self.hidden_dim, self.output_cfg['NUM_CAM_MAP'], kernel_size=1)
        self.output_proj3 = nn.Conv2d(self.hidden_dim, self.output_cfg['NUM_PARAMS_MAP'], kernel_size=1)

        self.bn1 = nn.BatchNorm2d(self.hidden_dim, momentum=BN_MOMENTUM)
        self.bn2 = nn.BatchNorm2d(self.hidden_dim, momentum=BN_MOMENTUM)
        self.bn3 = nn.BatchNorm2d(self.hidden_dim, momentum=BN_MOMENTUM)

        self.relu = nn.ReLU(inplace=True)

        if args().model_return_loss:
            self._calc_loss = Loss()
            
        if not args().fine_tune and not args().eval:
            self.init_weights()
            # self.backbone.load_pretrain_params()
            
            
    @property
    def hidden_dim(self):
        """ Returns the hidden feature dimension size. """
        return self.transformer.d_model  
    
    def _build_model(self):
        params_num, cam_dim = self._result_parser.params_map_parser.params_num, 3
        self.output_cfg = {'NUM_PARAMS_MAP':params_num-cam_dim, 'NUM_CENTER_MAP':1, 'NUM_CAM_MAP':cam_dim}
        self.transformer = build_transformer(args)

    def head_forward(self,meta_data, window_meta_data):
        """ The forward expects a NestedTensor, which consists of:
               - samples.tensors: batched images, of shape [batch_size x 3 x H x W]
               - samples.mask: a binary mask of shape [batch_size x H x W],
                               containing 1 on padded pixels

            It returns a dict with the following elements:
               - "Center Heatmap": Shape = [batch_size x num_queries x (num_classes + 1)]
               - "SMPL Parameters": Shape = (center_x, center_y, height, width)
        """
        # print(meta_data.shape,window_meta_data.shape, 111) 
        meta_data = meta_data.contiguous().cuda()
        window_meta_data = window_meta_data.contiguous().cuda()

        # meta_data = meta_data.permute(0,3,1,2) 
        # src = self.backbone(meta_data)['2'] # This is for Faster-RCNN

        src = self.backbone(meta_data)
        src = self.pool(src)   
        # position encoding        
        pos = self.position_embedding(src)

        # print(self.position_embedding.row_embed.weight,4444)

        # print(src.shape,pos.shape,222)
        
        # window_meta_data = window_meta_data.permute(0,3,1,2)
        # window_src = self.backbone(window_meta_data)['2']  # This is for Faster-RCNN
        window_src = self.backbone(window_meta_data)  
        window_src = self.pool(window_src)
        window_pos = self.position_embedding(window_src)

        # print(src.shape, window_src.shape, pos.shape, window_pos.shape, 333)   
        
        hs, hs_without_norm, memory = self.transformer(src, pos, window_src, window_pos)
            
        # print(hs.shape, memory.shape, 444)

        center_maps = self.bn1(hs)
        center_maps = self.relu(center_maps)
        center_maps = self.output_proj1(center_maps)
        # print(center_maps.shape,'c3')

        cam_maps = self.bn2(hs)
        cam_maps = self.relu(cam_maps)
        cam_maps = self.output_proj2(cam_maps)
        # print(cam_maps.shape,'cam')

        params_maps = self.bn3(hs)
        params_maps = self.relu(params_maps)
        params_maps = self.output_proj3(params_maps)
        # print(params_maps.shape,'params')

        # print(center_maps.shape,cam_maps.shape,params_maps.shape,555)
        
        # to make sure that scale is always a positive value
        cam_maps[:, 0] = torch.pow(1.1,cam_maps[:, 0])
        params_maps = torch.cat([cam_maps, params_maps], 1)
        output = {'params_maps':params_maps.float(), 
                  'center_map':center_maps.float()} 
        
        # print(output['center_map'].shape,output['params_maps'].shape,666)     
        # print(output['center_map'],666)  
        return output

if __name__ == '__main__':
    args().configs_yml = 'configs/v1.yml'
    args().model_version=1
    from models.build import build_model
    model = build_model().cuda()
    outputs=model.feed_forward({'image':torch.rand(4,512,512,3).cuda()})
    for key, value in outputs.items():
        if isinstance(value,tuple):
            print(key, value)
        elif isinstance(value,list):
            print(key, value)
        else:
            print(key, value.shape)