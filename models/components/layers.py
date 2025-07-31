import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional
from models.components.attentions import MultiHeadAttention, FeedForward
from utility.functions import create_causal_mask, create_padding_mask, create_cross_attention_mask


class EncoderLayer(nn.Module):
    def __init__(self, 
                 embed_dim, 
                 num_heads, 
                 dropout=0.1,
                 ):
        super(EncoderLayer, self).__init__()
        self.self_attention = MultiHeadAttention(embed_dim, num_heads) 
        self.feed_forward = FeedForward(embed_dim)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self, 
        x: Tensor,
        mask: Optional[Tensor] = None,
        padding_mask: Optional[Tensor] = None,
        is_causal: Optional[Tensor] = False
    ) -> Tensor:
        # Self-attention con connessione residua
        # x = x.transpose(0,1)
        attn_output = self.self_attention(
            x,
            x,
            x,
            mask,
            padding_mask,
            is_causal = is_causal
            )
        x = self.norm1(x + self.dropout(attn_output))

        # Feed-forward con connessione residua
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))

        return x

class DecoderLayer(nn.Module):
    def __init__(self, 
                 embed_dim, 
                 num_heads, 
                 dropout=0.1,
                 ):
        super(DecoderLayer, self).__init__()
                
        self.self_attention = MultiHeadAttention(embed_dim, num_heads) 
        self.cross_attention = MultiHeadAttention(embed_dim, num_heads) 
        self.feed_forward = FeedForward(embed_dim)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.norm3 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self,
                x,
                encoder_output,
                self_padding_mask = None, # Target
                cross_padding_mask = None, # Source
                ):
        # Self-attention mascherata
        attn_output = self.self_attention(
            x,
            x,
            x,
            None,
            is_causal = True
            )
        x = self.norm1(x + self.dropout(attn_output))

        # Cross-attention con l'encoder
        cross_attn_output = self.cross_attention(
            x,
            encoder_output, 
            encoder_output, 
            cross_padding_mask,
            is_causal = False
            )
        x = self.norm2(x + self.dropout(cross_attn_output))

        # Feed-forward
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_output))

        return x


    
if __name__ == '__main__':



    embed_dim = 128
    num_heads = 8
    hidden_dim = 200
    max_len = 5000
    batch_size = 32
    d_ff = hidden_dim
    num_head = 8
    
    la_tensor = torch.tensor(
        [[ 1,3,4,2,0,0,0],
         [ 1,3,4,4,2,0,0],
         [ 1,2,0,0,0,0,0]
         ]
    )
    
    src_padding_mask = create_padding_mask(la_tensor)
    print(f"{la_tensor.shape=}")
    print(f"{src_padding_mask=}")
    
    # en_tensor = torch.randint(1, 10**4,[batch_size, seq_len], dtype=int)
    en_tensor = torch.tensor(
        [[ 1,3,2,0,0,0,0],
         [ 1,3,3,3,2,0,0],
         [ 1,3,2,0,0,0,0]
         ]
    )
    tgt_padding_mask = create_padding_mask(en_tensor)
    print(f"{en_tensor.shape=}")
    print(f"{tgt_padding_mask=}")
    
    embedding = nn.Embedding(10**4, embedding_dim=embed_dim)
    la_embedding = embedding(la_tensor)
    en_embedding = embedding(en_tensor)
    
    encoder_layer = EncoderLayer(
        embed_dim=embed_dim,
        num_heads=num_head,
    )    
    
    decoder_layer = DecoderLayer(
        embed_dim=embed_dim,
        num_heads=num_head,
    )  

    built_encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_head
    )

    built_decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim,
            nhead=num_head
    )

    memory = built_encoder_layer(la_embedding.transpose(0,1), src_key_padding_mask=( la_tensor == 0 ))
    
    print(built_decoder_layer(en_embedding.transpose(0,1), memory,
                              tgt_key_padding_mask=(tgt_padding_mask == False),
                              memory_key_padding_mask=(src_padding_mask == False)
                              ))
    
    
    print(
        f"\n{la_tensor.shape=}"
        f"\n{la_embedding.shape=}",
        f"\n{encoder_layer(la_embedding).shape=}",
        f"\n{decoder_layer(en_embedding, encoder_layer(la_embedding)).shape=}"
    )
    del la_embedding, en_embedding, encoder_layer, decoder_layer, embedding