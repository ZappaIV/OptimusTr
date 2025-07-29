import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from transformers import BertLayer, BertConfig

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

    def forward(self, x, padding_mask=None):
        # Embedding + scaling + positional encoding
        x = self.embedding(x) * math.sqrt(self.embed_dim)
        x = self.positional_encoding(x)
        x = self.dropout(x)

        # Passa attraverso tutti i layer dell'encoder
        for layer in self.layers:
            x = layer(x, padding_mask=padding_mask)

        return x

if __name__ == '__main__':

    embed_dim = 128
    num_heads = 8
    hidden_dim = 200
    max_len = 5000
    batch_size = 32
    d_ff = hidden_dim
    num_head = 8
    
    la_seq_len = 9
    
    la_tensor = torch.randint(1, 10**4,[batch_size, la_seq_len], dtype=int)
    print(f"{la_tensor.shape=}")
    
    encoder = Encoder(
        vocab_size = 10**4,
        embed_dim=embed_dim,
        num_heads=num_head,
        n_layers=2,
        max_len=5000,
    )
    
    print(encoder)
    print(
        f"\n{encoder(la_tensor).shape=}",
    )
    del encoder