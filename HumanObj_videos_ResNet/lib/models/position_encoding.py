# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Various positional encodings for the transformer.
"""
import math
import torch
from torch import nn

from utils.misc import NestedTensor



class PositionEmbeddingLearned(nn.Module):
    """
    Absolute pos embedding, learned.
    """
    def __init__(self, num_pos_feats=256):
        super().__init__()
        self.row_embed = nn.Embedding(128, num_pos_feats) # the first dimension of the embedding should at least be image height
                                                          # Backbone output is of size b, c, 128, 128
        self.col_embed = nn.Embedding(128, num_pos_feats)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.row_embed.weight)
        nn.init.uniform_(self.col_embed.weight)

    def forward(self, x):
        h, w = x.shape[-2:]
        i = torch.arange(w, device=x.device)
        j = torch.arange(h, device=x.device)

        x_emb = self.col_embed(i)
        y_emb = self.row_embed(j)
        
        pos = torch.cat([
            x_emb.unsqueeze(0).repeat(h, 1, 1),
            y_emb.unsqueeze(1).repeat(1, w, 1),
        ], dim=-1).permute(2, 0, 1).unsqueeze(0).repeat(x.shape[0], 1, 1, 1)
                
        return pos


def build_position_encoding(args):
    n_steps = args().hidden_dim_pos // 2
    
    if args().position_embedding=='learned':
        position_embedding = PositionEmbeddingLearned(n_steps)
    else:
        raise ValueError(f"not supported {args.position_embedding}")

    return position_embedding
