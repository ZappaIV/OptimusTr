import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from models.components.embeddings import PositionalEncoding
from models.components.layers import EncoderLayer

class Encoder(nn.Module):
    def __init__(self,
                 vocab_size,
                 embed_dim, num_heads, 
                 d_ff,
                 n_layers,
                 max_len=5000,
                 dropout = 0.1,
                 use_nn_mha=False,
                 ):
        super(Encoder, self).__init__()
        self.embed_dim = embed_dim
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.positional_encoding = PositionalEncoding(embed_dim, max_len)
        self.layers = nn.ModuleList([
            EncoderLayer(embed_dim, num_heads, d_ff, dropout, use_nn_mha) 
            for _ in range(n_layers)
            ])
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # Embedding + scaling + positional encoding
        x = self.embedding(x) * math.sqrt(self.embed_dim)
        x = self.positional_encoding(x)
        x = self.dropout(x)

        # Passa attraverso tutti i layer dell'encoder
        for layer in self.layers:
            x = layer(x, mask)

        return x

if __name__ == '__main__':

    embed_dim = 128
    num_heads = 8
    hidden_dim = 200
    max_len = 5000
    batch_size = 32
    d_ff = hidden_dim
    num_head = 8
    
    la_tensor = torch.ones(batch_size, embed_dim, dtype=int)
    print(la_tensor.shape)
    
    encoder = Encoder(
        vocab_size = 10**4,
        embed_dim=embed_dim,
        num_heads=num_head,
        d_ff=d_ff,
        n_layers=2,
        max_len=5000,
        use_nn_mha=True
    )
    
    print(encoder)
    print(
        f"\n{encoder(la_tensor).shape=}",
    )
    del encoder