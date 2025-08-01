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
        tgt_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None, # Target x Target
        memory_mask: Optional[Tensor] = None, 
        memory_key_padding_mask: Optional[Tensor] = None, # Source x Target
        tgt_is_causal: Optional[Tensor] = None,
        memory_is_causal: Optional[Tensor] = None
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
                tgt_mask = tgt_mask, 
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_mask=memory_mask,
                memory_key_padding_mask=memory_key_padding_mask,
                tgt_is_causal = tgt_is_causal,
                memory_is_causal = memory_is_causal
            )

        return x
