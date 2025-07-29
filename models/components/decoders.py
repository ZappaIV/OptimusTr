import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from transformers import BertLayer, BertConfig

from models.components.embeddings import PositionalEncoding
from models.components.layers import DecoderLayer

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

    def forward(self, 
                x,
                encoder_output,
                self_padding_mask=None,
                cross_padding_mask=None
                ):
        
        # Embedding + scaling + positional encoding
        x = self.embedding(x) * math.sqrt(self.embed_dim)
        x = self.positional_encoding(x)
        x = self.dropout(x)

        # Passa attraverso tutti i layer del decoder
        for layer in self.layers:
            x = layer(x, 
                      encoder_output, 
                      self_padding_mask=self_padding_mask, 
                      cross_padding_mask=cross_padding_mask
                      )

        return x

if __name__ == '__main__':
    
    from models.components.encoders import Encoder
    from models.components.attentions import create_padding_mask

    embed_dim = 128
    num_heads = 8
    hidden_dim = 200
    max_len = 5000
    batch_size = 4
    d_ff = hidden_dim
    num_head = 8
    
    seq_len = 9
    seq_len = 10
    
    # la_tensor = torch.randint(1, 10**4,[batch_size, seq_len], dtype=int)
    
    la_tensor = torch.tensor(
        [[ 1,3,4,2,0,0,0],
         [ 1,3,4,4,2,0,0],
         [ 1,2,0,0,0,0,0]]
    )
    
    src_padding_mask = create_padding_mask(la_tensor)
    print(f"{la_tensor.shape=}")
    print(f"{src_padding_mask=}")
    
    # en_tensor = torch.randint(1, 10**4,[batch_size, seq_len], dtype=int)
    en_tensor = torch.tensor(
        [[ 1,3,2,0,0,0,0],
         [ 1,3,3,3,2,0,0],
         [ 1,3,2,0,0,0,0]]
    )
    tgt_padding_mask = create_padding_mask(en_tensor)
    print(f"{en_tensor.shape=}")
    print(f"{tgt_padding_mask=}")
    
    encoder = Encoder(
        vocab_size = 10**4,
        embed_dim=embed_dim,
        num_heads=num_head,
        n_layers=2,
        max_len=5000,
    )
    
    decoder = Decoder(
        vocab_size = 10**4,
        embed_dim=embed_dim,
        num_heads=num_head,
        n_layers=2,
        max_len=5000,
    )

    print(encoder)
    print(decoder)
    
    encoder_output = encoder(
            la_tensor, 
            padding_mask=src_padding_mask
            )

    
    decoder_output = decoder(
        en_tensor,
        encoder_output,
        self_padding_mask=tgt_padding_mask,
        cross_padding_mask=src_padding_mask
            )
    
    print(
        f"\n{encoder_output.shape=}",
        f"\n{decoder_output.shape=}"
    )
    del encoder, decoder