import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch import Tensor
from transformers import BertLayer, BertConfig
from typing import Optional 

from models.components.embeddings import PositionalEncoding
from models.components.layers import EncoderLayer

class Encoder(nn.Module):
    def __init__(self,
                 vocab_size,
                 embed_dim, num_heads, 
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
        mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
        is_causal: Optional[bool] = False
    ) -> Tensor:
        # Embedding + scaling + positional encoding
        x = self.embedding(x) * math.sqrt(self.embed_dim)
        x = self.positional_encoding(x)
        x = self.dropout(x)

        # Passa attraverso tutti i layer dell'encoder
        for layer in self.layers:
            x = layer(
                x,
                mask = mask,
                padding_mask=src_key_padding_mask,
                is_causal = is_causal
            )

        return x

if __name__ == '__main__':

    from models.components.attentions import create_padding_mask

    embed_dim = 128
    num_heads = 8
    hidden_dim = 200
    max_len = 5000
    batch_size = 32
    d_ff = hidden_dim
    num_head = 8
    
    la_seq_len = 9
    
    la_tensor = torch.tensor(
        [[ 1,3,4,2,0,0,0],
         [ 1,3,4,4,2,0,0],
         [ 1,2,0,0,0,0,0]]
    )
    
    src_padding_mask = create_padding_mask(la_tensor)
    print(f"{la_tensor.shape=}")
    print(f"{src_padding_mask=}")
    
    encoder = Encoder(
        vocab_size = 10**4,
        embed_dim=embed_dim,
        num_heads=num_head,
        n_layers=2,
        max_len=5000,
    )
    
    print(encoder)
    memory = encoder(
        la_tensor,
        src_padding_mask=src_padding_mask
    )
    
    print(
        f"\n{memory.shape=}",
    )
    del encoder