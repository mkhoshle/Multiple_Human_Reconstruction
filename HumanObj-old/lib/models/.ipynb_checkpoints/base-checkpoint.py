from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os,sys
import torch
import torch.nn as nn

from utils.misc import NestedTensor, nested_tensor_from_tensor_list

root_dir = os.path.join(os.path.dirname(__file__),'..')
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)
import config
from config import args
from utils import print_dict
if args().model_precision=='fp16':
    from torch.cuda.amp import autocast

BN_MOMENTUM = 0.1
default_cfg = {'mode':'val', 'calc_loss': False}

class Base(nn.Module):
    def forward(self, meta_data, window_meta_data, **cfg):
        if cfg['mode'] == 'matching_gts':
            return self.matching_forward(meta_data, window_meta_data, **cfg)
        elif cfg['mode'] == 'parsing':
            return self.parsing_forward(meta_data, window_meta_data, **cfg)
        elif cfg['mode'] == 'forward':
            return self.pure_forward(meta_data, window_meta_data, **cfg)
        else:
            raise NotImplementedError('forward mode is not recognized! please set proper mode (parsing/matching_gts)')

    def matching_forward(self, meta_data, window_meta_data, **cfg):
        meta_data['image'] = meta_data['image'].contiguous().cuda()
        window_meta_data = window_meta_data.contiguous().cuda()
        if args().model_precision=='fp16':
            with autocast():
                outputs = self.feed_forward(meta_data, window_meta_data) 
                outputs, meta_data = self._result_parser.matching_forward(outputs, meta_data, cfg)
        else:
            outputs = self.feed_forward(meta_data, window_meta_data)
            outputs, meta_data = self._result_parser.matching_forward(outputs, meta_data, cfg)
                
        outputs['meta_data'] = meta_data
        if cfg['calc_loss']:
            outputs.update(self._calc_loss(outputs))

        return outputs

    @torch.no_grad()
    def parsing_forward(self, meta_data, window_meta_data, **cfg):
        meta_data['image'] = meta_data['image'].contiguous().cuda()
        window_meta_data = window_meta_data.contiguous().cuda()
        if args().model_precision=='fp16':
            with autocast():
                print(meta_data['image'].shape,window_meta_data.shape,66)
                outputs = self.feed_forward(meta_data, window_meta_data)
                print(55)
                outputs, meta_data = self._result_parser.parsing_forward(outputs, meta_data, cfg)
                print(outputs.keys(),444)
        else:
            outputs = self.feed_forward(meta_data, window_meta_data)
            outputs, meta_data = self._result_parser.parsing_forward(outputs, meta_data, cfg)

        outputs['meta_data'] = meta_data
        print(outputs.keys(),555)
        return outputs

    def feed_forward(self, meta_data, window_meta_data): 
        # print(meta_data['image'].shape,window_meta_data.shape,22)
        outputs = self.head_forward(meta_data['image'], window_meta_data)
        return outputs

    @torch.no_grad()
    def pure_forward(self, meta_data, window_meta_data, **cfg):
        if args().model_precision=='fp16':
            with autocast():
                # print(meta_data['image'].shape,window_meta_data.shape,44)
                outputs = self.feed_forward(meta_data, window_meta_data)
        else:
            outputs = self.feed_forward(meta_data, window_meta_data)
        return outputs

    def head_forward(self,x1,x2):
        return NotImplementedError

    def make_backbone(self):
        return NotImplementedError

    def backbone_forward(self, x):
        return NotImplementedError

    def _build_gpu_tracker(self):
        self.gpu_tracker = MemTracker()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.001)
                for name, _ in m.named_parameters():
                    if name in ['bias']:
                        nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.ConvTranspose2d):
                nn.init.normal_(m.weight, std=0.001)
                for name, _ in m.named_parameters():
                    if name in ['bias']:
                        nn.init.constant_(m.bias, 0)