# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Misc functions, including distributed helpers.

Mostly copy-paste from torchvision references.
"""
import os
import pickle
from collections import defaultdict, deque
from typing import List, Optional

import torch
import torch.distributed as dist
import torch.nn.functional as F
import torch.distributed as dist
# needed due to empty tensor bug in pytorch and torchvision 0.5
import torchvision
from torch import Tensor

def is_main_process():
    return get_rank() == 0

def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()

def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def _max_by_axis(the_list):
    # type: (List[List[int]]) -> List[int]
    maxes = the_list[0]
    for sublist in the_list[1:]:
        for index, item in enumerate(sublist):
            maxes[index] = max(maxes[index], item)
    return maxes

def nested_tensor_from_tensor_list(tensor):
    # TODO make this more general
    if isinstance(tensor,list):
        if tensor_list[0].ndim == 3:
            # TODO make it support different-sized images
            max_size = _max_by_axis([list(img.shape) for img in tensor_list])

            # min_size = tuple(min(s) for s in zip(*[img.shape for img in tensor_list]))
            batch_shape = [len(tensor_list)] + max_size
            b, _, h, w = batch_shape
            dtype = tensor_list[0].dtype
            device = tensor_list[0].device

            tensor = torch.zeros(batch_shape, dtype=dtype, device=device)
            mask = torch.ones((b, h, w), dtype=torch.bool, device=device)
            for img, pad_img, m in zip(tensor_list, tensor, mask):
                pad_img[: img.shape[0], : img.shape[1], : img.shape[2]].copy_(img)
                m[: img.shape[1], :img.shape[2]] = False      
        else:
            raise ValueError('not supported')
    else:
        b, _, h, w = tensor.shape
        mask = torch.ones((b, h, w), dtype=torch.bool, device=tensor.device)
        for m in mask:
            m[: h, :w] = False 
        
    return NestedTensor(tensor, mask)


class NestedTensor(object):
    def __init__(self, tensors, mask: Optional[Tensor] = None):
        self.tensors = tensors
        self.mask = mask

    def to(self, device):
        # type: (Device) -> NestedTensor # noqa
        cast_tensor = self.tensors.to(device)
        mask = self.mask
        if mask is not None:
            assert mask is not None
            cast_mask = mask.to(device)
        else:
            cast_mask = None
        return NestedTensor(cast_tensor, cast_mask)

    def decompose(self):
        return self.tensors, self.mask

    def __repr__(self):
        return str(self.tensors)