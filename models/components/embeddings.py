import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    def __init__(self, 
                 embed_dim, 
                 max_len=5000
                 ):
        super(PositionalEncoding, self).__init__()

        pe = torch.zeros(max_len, embed_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)

        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]

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

    positional_encoding = PositionalEncoding(
        embed_dim=embed_dim,
        max_len=max_len
    )   
    
    embedding = nn.Embedding(10**4, embedding_dim=embed_dim)
    la_embedding = embedding(la_tensor)
    
    print(
        f"\n{la_tensor.shape=}"
        f"\n{la_embedding.shape=}",
        f"\n{positional_encoding(la_embedding).shape=}"
        )
    
    del la_embedding, embedding, positional_encoding
    