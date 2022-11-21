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
from models.basic_modules import BasicBlock,Bottleneck
from models.transformer import build_transformer

import config
from config import args
from loss_funcs import Loss
from maps_utils.result_parser import ResultParser
from utils.misc import NestedTensor, nested_tensor_from_tensor_list

BN_MOMENTUM = 0.1


class HOBJ(Base):
    def __init__(self, backbone=None, **kwargs):
        super(HOBJ, self).__init__()
        print('Using HOBJ v1')    
        self.backbone = backbone
        self._result_parser = ResultParser()
        self._build_model()
        self.input_proj = nn.Conv2d(self.backbone.num_channels[-1], self.hidden_dim, kernel_size=1)
        self.output_proj1 = nn.Conv2d(self.hidden_dim, self.output_cfg['NUM_CENTER_MAP'], kernel_size=1)
        self.output_proj2 = nn.Conv2d(self.hidden_dim, self.output_cfg['NUM_CAM_MAP'], kernel_size=1)
        self.output_proj3 = nn.Conv2d(self.hidden_dim, self.output_cfg['NUM_PARAMS_MAP'], kernel_size=1)

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
        if not isinstance(meta_data, NestedTensor):
            meta_data = nested_tensor_from_tensor_list(meta_data)  
        
        # print(meta_data.tensors.shape,111) 
        features, pos = self.backbone(meta_data)  
        src, mask = features[-1].decompose()  
                
        # print(src.shape,222)
        src = self.input_proj(src)     
        pos = pos[-1]
        
        # print(src.shape,pos.shape,333)
        if not isinstance(window_meta_data, NestedTensor):
            window_meta_data = nested_tensor_from_tensor_list(window_meta_data) 
         
        window_features, window_pos = self.backbone(window_meta_data)
        window_src, _ = window_features[-1].decompose()    
        window_src = self.input_proj(window_src)
        window_pos = torch.stack([pos[-1] for pos in window_pos])
       
    
        batch_size, _, h, w = src.shape
        hs, hs_without_norm, memory = self.transformer(src, pos, window_src, window_pos)
            
        # print(hs.shape,444)
        
        center_maps = self.output_proj1(hs)
        cam_maps = self.output_proj2(hs)
        params_maps = self.output_proj3(hs)
        
        # print(center_maps.shape,cam_maps.shape,params_maps.shape,555)
        
        # to make sure that scale is always a positive value
        cam_maps[:, 0] = torch.pow(1.1,cam_maps[:, 0])
        params_maps = torch.cat([cam_maps, params_maps], 1)
        output = {'params_maps':params_maps.float(), 
                  'center_map':center_maps.float()} 
        
        # print(output['center_map'].shape,output['params_maps'].shape,666)                
        return output

    def corr2d_multi_in_out_1x1(X, K):
        c_i, h, w = X.shape
        c_o = K.shape[0]
        X = X.reshape((c_i, h * w))
        K = K.reshape((c_o, c_i))
        # Matrix multiplication in the fully-connected layer
        Y = torch.matmul(K, X)
        return Y.reshape((c_o, h, w))
        
            
if __name__ == '__main__':
    args().configs_yml = 'configs/v1.yml'
    args().model_version=1
    from models.build import build_model
    model = build_model().cuda()
    outputs = model.feed_forward({'image':torch.rand(4,512,512,3).cuda()})
    for key, value in outputs.items():
        if isinstance(value,tuple):
            print(key, value)
        elif isinstance(value,list):
            print(key, value)
        else:
            print(key, value.shape)