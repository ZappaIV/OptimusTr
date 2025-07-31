import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from transformers import BertLayer, BertConfig
from torch import Tensor
from typing import Optional

from src.transformers.models.embeddings import PositionalEncoding
from src.transformers.models.layers import DecoderLayer

class Decoder(nn.Module):
    def __init__(self,
                 vocab_size,
                 embed_dim, 
                 num_heads,
                 n_layers, 
                 max_len=5000,
                 dropout=0.1,
                 ):
        super(Decoder, self).__init__()
        self.embed_dim = embed_dim
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.positional_encoding = PositionalEncoding(embed_dim, max_len)
        self.layers = nn.ModuleList([
            DecoderLayer(embed_dim, num_heads, dropout) 
            for _ in range(n_layers)
            ])
        self.dropout = nn.Dropout(dropout)

    def forward( 
        self,
        x: Tensor,
        memory: Tensor,
        mask: Optional[Tensor] = None,
        self_padding_mask: Optional[Tensor] = None, # Target x Target
        cross_padding_mask: Optional[Tensor] = None, # Source x Target
        self_is_causal: Optional[Tensor] = None,
        cross_is_causal: Optional[Tensor] = None
    ):
        
        # Embedding + scaling + positional encoding
        x = self.embedding(x) * math.sqrt(self.embed_dim)
        x = self.positional_encoding(x)
        x = self.dropout(x)

        # Passa attraverso tutti i layer del decoder
        for layer in self.layers:
            x = layer(
                x, 
                memory,
                mask = mask, 
                self_padding_mask=self_padding_mask, 
                cross_padding_mask=cross_padding_mask,
                self_is_causal = self_is_causal,
                cross_is_causal = cross_is_causal
            )

        return x
