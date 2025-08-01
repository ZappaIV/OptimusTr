import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch import Tensor
from transformers import BertLayer, BertConfig
from typing import Optional 

from src.transformers.models.embeddings import PositionalEncoding
from src.transformers.models.layers import EncoderLayer

class Encoder(nn.Module):
    def __init__(
        self,
        vocab_size,
        embed_dim, 
        num_heads, 
        n_layers,
        max_len=5000,
        dropout = 0.1,
    ):
        super(Encoder, self).__init__()
        self.embed_dim = embed_dim
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.positional_encoding = PositionalEncoding(embed_dim, max_len)
        self.layers = nn.ModuleList([
            EncoderLayer(embed_dim, num_heads, dropout) 
            for _ in range(n_layers)
            ])
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: Tensor,
        src_mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
        src_is_causal: Optional[Tensor] = False
    ) -> Tensor:
        # Embedding + scaling + positional encoding
        x = self.embedding(x) * math.sqrt(self.embed_dim)
        x = self.positional_encoding(x)
        x = self.dropout(x)

        # Passa attraverso tutti i layer dell'encoder
        for layer in self.layers:
            x = layer(
                x,
                src_mask = src_mask,
                src_key_padding_mask=src_key_padding_mask,
                src_is_causal = src_is_causal
            )

        return x
